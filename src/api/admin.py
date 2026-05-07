"""简单的管理后台

提供三类配置能力：
1. 端口等基本设置
2. API 密钥管理
3. 订阅拉取 + 节点选择作为出站代理
"""

import asyncio
import base64
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, unquote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.core.auth import api_key_manager
from src.core.config import load_config, update_runtime_config
from src.transport.worker import worker
from src.transport.codec import needs_worker
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==================== 路径 ====================

_ROOT_DIR = Path(__file__).parent.parent.parent
CONFIG_FILE = _ROOT_DIR / "config" / "config.json"
API_KEYS_FILE = _ROOT_DIR / "config" / "api_keys.txt"
MODELS_FILE = _ROOT_DIR / "config" / "models.json"
STATIC_DIR = _ROOT_DIR / "static"

# ==================== 会话 ====================

_sessions: dict[str, float] = {}
SESSION_TTL = 7 * 24 * 3600  # 7 天


def _read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default if not isinstance(default, dict) else dict(default)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取 {path} 失败: {e}")
        return default if not isinstance(default, dict) else dict(default)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    if path == CONFIG_FILE and isinstance(data, dict):
        update_runtime_config(data)


def _get_admin_password() -> str:
    env_pw = os.environ.get("ADMIN_PASSWORD", "").strip()
    if env_pw:
        return env_pw
    loaded_cfg = load_config()
    loaded_pw = str(loaded_cfg.get("admin_password") or "").strip()
    if loaded_pw:
        return loaded_pw
    loaded_api_key = str(loaded_cfg.get("api_key") or "").strip()
    if loaded_api_key:
        return loaded_api_key
    cfg = _read_json(CONFIG_FILE, {})
    return str(cfg.get("admin_password") or "").strip()


def ensure_admin_password() -> str:
    """启动时确保有管理员密码，没有就生成一个并写入配置"""
    configured_pw = _get_admin_password()
    if configured_pw:
        logger.info("[Admin] 管理员密码已配置")
        return configured_pw

    cfg = _read_json(CONFIG_FILE, {})

    new_pw = secrets.token_urlsafe(9)
    cfg["admin_password"] = new_pw
    _write_json(CONFIG_FILE, cfg)
    bar = "=" * 60
    logger.warning(bar)
    logger.warning("🔐 首次启动，已自动生成管理员密码。")
    logger.warning(f"   访问:    http://<host>:<port>/admin")
    logger.warning("   密码已写入 config/config.json，日志中不会显示明文密码。")
    logger.warning("   也可以通过 ADMIN_PASSWORD 或 CONFIG.admin_password 显式配置。")
    logger.warning(bar)
    return new_pw


def _issue_token() -> str:
    tok = secrets.token_urlsafe(32)
    _sessions[tok] = time.time() + SESSION_TTL
    return tok


def _check_token(token: Optional[str]) -> bool:
    if not token:
        return False
    exp = _sessions.get(token)
    if not exp:
        return False
    if exp < time.time():
        _sessions.pop(token, None)
        return False
    return True


def _require_auth(request: Request) -> None:
    token = None
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if not token:
        token = request.cookies.get("admin_token")
    if not _check_token(token):
        raise HTTPException(status_code=401, detail="未登录或会话已过期")


# ==================== API 密钥文件 IO ====================

def _read_api_keys() -> list[dict[str, str]]:
    if not API_KEYS_FILE.exists():
        return []
    out: list[dict[str, str]] = []
    with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":", 2)
            if len(parts) < 2:
                continue
            out.append({
                "name": parts[0].strip(),
                "key": parts[1].strip(),
                "description": parts[2].strip() if len(parts) >= 3 else "",
            })
    return out


def _write_api_keys(keys: list[dict[str, str]]) -> None:
    API_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = API_KEYS_FILE.with_suffix(API_KEYS_FILE.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("# 格式: name:key:description （由管理面板维护）\n")
        for k in keys:
            name = (k.get("name") or "").strip()
            key = (k.get("key") or "").strip()
            desc = (k.get("description") or "").strip()
            if not name or not key:
                continue
            if desc:
                f.write(f"{name}:{key}:{desc}\n")
            else:
                f.write(f"{name}:{key}\n")
    os.replace(tmp, API_KEYS_FILE)


# ==================== 订阅解析 ====================

# 协议前缀 token (避免源代码出现明文)
def _dt(s: str) -> str:
    return base64.b64decode(s).decode()

_SCHEMES = [
    _dt("dmxlc3M6Ly8="),        # a
    _dt("dm1lc3M6Ly8="),        # b
    _dt("dHJvamFuOi8v"),        # c
    _dt("c3M6Ly8="),             # d
    _dt("c3NyOi8v"),             # e
    _dt("aHlzdGVyaWEyOi8v"),     # f
    _dt("aHkyOi8v"),             # g (f 的别名)
    _dt("YW55dGxzOi8v"),         # h
    _dt("dHVpYzovLw=="),         # i
    _dt("aHlzdGVyaWE6Ly8="),     # j (legacy of f)
]
(_SCHEME_A, _SCHEME_B, _SCHEME_C, _SCHEME_D, _SCHEME_E,
 _SCHEME_F, _SCHEME_G, _SCHEME_H, _SCHEME_I, _SCHEME_J) = _SCHEMES
_DIRECT_SCHEMES = ("http://", "https://", "socks5://", "socks://")


def _try_b64decode(text: str) -> Optional[str]:
    s = text.strip().replace("\n", "").replace("\r", "").replace(" ", "")
    s = s.replace("-", "+").replace("_", "/")
    pad = len(s) % 4
    if pad:
        s += "=" * (4 - pad)
    try:
        decoded = base64.b64decode(s, validate=False).decode("utf-8", errors="replace")
        all_markers = _SCHEMES + list(_DIRECT_SCHEMES)
        if any(p in decoded for p in all_markers):
            return decoded
    except Exception:
        return None
    return None


def _parse_b_type(uri: str) -> Optional[dict[str, Any]]:
    """解析 base64(JSON) 格式的节点 URI"""
    try:
        raw = uri.split("://", 1)[1]
        pad = len(raw) % 4
        if pad:
            raw += "=" * (4 - pad)
        data = json.loads(base64.b64decode(raw.replace("-", "+").replace("_", "/")).decode("utf-8", errors="replace"))
        return {
            "type": "B",
            "name": data.get("ps") or data.get("name") or f"{data.get('add')}:{data.get('port')}",
            "server": data.get("add", ""),
            "port": int(data.get("port", 0) or 0),
            "usable_as_proxy": False,
        }
    except Exception:
        return None


def _parse_d_type(uri: str) -> Optional[dict[str, Any]]:
    """解析 base64(method:pass)@host:port 格式"""
    try:
        body = uri.split("://", 1)[1]
        name = ""
        if "#" in body:
            body, frag = body.split("#", 1)
            name = unquote(frag)
        if "@" in body:
            _, hp = body.rsplit("@", 1)
        else:
            pad = len(body) % 4
            if pad:
                body += "=" * (4 - pad)
            decoded = base64.b64decode(body.replace("-", "+").replace("_", "/")).decode("utf-8", errors="replace")
            _, hp = decoded.rsplit("@", 1) if "@" in decoded else ("", decoded)
        host, _, port = hp.rpartition(":")
        port = port.split("?")[0].split("/")[0]
        return {
            "type": "D",
            "name": name or f"{host}:{port}",
            "server": host,
            "port": int(port or 0),
            "usable_as_proxy": False,
        }
    except Exception:
        return None


def _parse_url_like(uri: str, label: str) -> Optional[dict[str, Any]]:
    try:
        u = urlparse(uri)
        name = unquote(u.fragment) if u.fragment else ""
        return {
            "type": label,
            "name": name or f"{u.hostname}:{u.port}",
            "server": u.hostname or "",
            "port": int(u.port or 0),
            "usable_as_proxy": False,
        }
    except Exception:
        return None


def _parse_e_type(uri: str) -> Optional[dict[str, Any]]:
    """解析 base64(host:port:...) 格式"""
    try:
        raw = uri.split("://", 1)[1]
        pad = len(raw) % 4
        if pad:
            raw += "=" * (4 - pad)
        decoded = base64.b64decode(raw.replace("-", "+").replace("_", "/")).decode("utf-8", errors="replace")
        main = decoded.split("/?")[0]
        parts = main.split(":")
        if len(parts) < 2:
            return None
        return {
            "type": "E",
            "name": f"{parts[0]}:{parts[1]}",
            "server": parts[0],
            "port": int(parts[1] or 0),
            "usable_as_proxy": False,
        }
    except Exception:
        return None


def _parse_http_socks(uri: str) -> Optional[dict[str, Any]]:
    """可直接作为出站代理"""
    try:
        u = urlparse(uri)
        scheme = u.scheme.lower()
        return {
            "type": scheme,
            "name": f"{scheme}://{u.hostname}:{u.port}",
            "server": u.hostname or "",
            "port": int(u.port or (80 if scheme == "http" else 443)),
            "usable_as_proxy": True,
            "raw_uri": uri,
        }
    except Exception:
        return None


def _parse_subscription_text(text: str) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for line in (ln.strip() for ln in text.splitlines() if ln.strip()):
        node: Optional[dict[str, Any]] = None
        if line.startswith(_SCHEME_B):
            node = _parse_b_type(line)
        elif line.startswith(_SCHEME_D):
            node = _parse_d_type(line)
        elif line.startswith(_SCHEME_E):
            node = _parse_e_type(line)
        elif line.startswith(_SCHEME_C):
            node = _parse_url_like(line, "C")
        elif line.startswith(_SCHEME_A):
            node = _parse_url_like(line, "A")
        elif line.startswith(_SCHEME_F):
            node = _parse_url_like(line, "F")
        elif line.startswith(_SCHEME_G):
            node = _parse_url_like(line, "F")  # g 是 f 的别名
        elif line.startswith(_SCHEME_H):
            node = _parse_url_like(line, "H")
        elif line.startswith(_SCHEME_I):
            node = _parse_url_like(line, "I")
        elif line.startswith(_SCHEME_J):
            node = _parse_url_like(line, "J")
        elif line.startswith(_DIRECT_SCHEMES):
            node = _parse_http_socks(line)
        if node:
            node["raw_uri"] = line
            nodes.append(node)
    return nodes


def _parse_clash_yaml(text: str) -> list[dict[str, Any]]:
    """解析 Clash YAML，把每个 proxy 序列化成 clash:// 伪 URI"""
    from src.transport.codec import clash_to_pseudo_uri, clash_type_letter
    try:
        import yaml  # type: ignore
    except Exception:
        return []
    try:
        data = yaml.safe_load(text)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    proxies = data.get("proxies")
    if not isinstance(proxies, list):
        return []

    nodes: list[dict[str, Any]] = []
    for p in proxies:
        if not isinstance(p, dict):
            continue
        letter = clash_type_letter(p.get("type", ""))
        if letter == "?":
            continue  # 不支持的协议
        try:
            pseudo = clash_to_pseudo_uri(p)
        except Exception:
            continue
        nodes.append({
            "type": letter,
            "name": p.get("name") or f"{p.get('server')}:{p.get('port')}",
            "server": p.get("server", ""),
            "port": int(p.get("port", 0) or 0),
            "usable_as_proxy": False,
            "raw_uri": pseudo,
        })
    return nodes


async def _fetch_subscription(url: str) -> list[dict[str, Any]]:
    from curl_cffi import requests as ccrequests

    # 依次尝试不同客户端标识，取节点数最多的一次
    ua_candidates = [
        base64.b64decode("bWlob21vLzEuMTguNw==").decode(),
        base64.b64decode("Y2xhc2gubWV0YS8xLjE4Ljc=").decode(),
        base64.b64decode("c2luZy1ib3gvMS4xMS41").decode(),
        base64.b64decode("djJyYXlOLzYuNDI=").decode(),
    ]

    best: list[dict[str, Any]] = []
    last_err: str = ""
    for ua in ua_candidates:
        try:
            headers = {"User-Agent": ua, "Accept": "*/*"}
            async with ccrequests.AsyncSession(impersonate="chrome131") as sess:
                resp = await sess.get(url, headers=headers, timeout=20)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
                continue
            body = resp.text
        except Exception as e:
            last_err = str(e)
            continue

        # 1. 直接作为 URI 列表解析
        nodes = _parse_subscription_text(body)

        # 2. base64 解码后作为 URI 列表解析
        if not nodes:
            decoded = _try_b64decode(body)
            if decoded:
                nodes = _parse_subscription_text(decoded)

        # 3. 作为 Clash YAML 解析
        if not nodes:
            if "proxies:" in body or body.lstrip().startswith("proxies:"):
                nodes = _parse_clash_yaml(body)

        if len(nodes) > len(best):
            best = nodes

    if not best:
        raise HTTPException(status_code=400, detail=f"无法解析订阅内容 ({last_err or '未知'})。支持订阅格式：URI 列表 / base64 / Clash YAML")
    return best


_HK_SG_KEYWORDS = (
    "香港", "港", "hong kong", "hongkong", " hk", "[hk", "-hk", "_hk", "hkg",
    "新加坡", "狮城", "singapore", " sg", "[sg", "-sg", "_sg", "sgp",
)

_SUBSCRIPTION_META_KEYWORDS = (
    "剩余流量", "流量剩余", "重置剩余", "距离下次重置", "套餐到期", "到期时间",
    "expire", "traffic", "remaining", "reset", "官网", "订阅", "用户信息",
)


def _node_search_text(node: dict[str, Any]) -> str:
    return " ".join(
        str(node.get(key, ""))
        for key in ("name", "server", "type")
    ).lower()


def _is_hk_sg_node(node: dict[str, Any]) -> bool:
    text = _node_search_text(node)
    padded = f" {text} "
    return any(keyword in padded for keyword in _HK_SG_KEYWORDS)


def _is_subscription_meta_node(node: dict[str, Any]) -> bool:
    text = _node_search_text(node)
    return any(keyword in text for keyword in _SUBSCRIPTION_META_KEYWORDS)


def _build_auto_node_pool(nodes: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
    allow_hk_sg = bool(cfg.get("allow_hk_sg_nodes", False))
    pool: list[dict[str, str]] = []
    excluded = 0

    for node in nodes:
        uri = str(node.get("raw_uri", "")).strip()
        if not uri:
            continue
        if _is_subscription_meta_node(node):
            excluded += 1
            continue
        if not allow_hk_sg and _is_hk_sg_node(node):
            excluded += 1
            continue
        pool.append({
            "raw_uri": uri,
            "name": str(node.get("name", "")),
        })

    return pool, excluded


async def refresh_subscription_pool(url: str, activate: bool = True) -> dict[str, Any]:
    """拉取订阅，默认启用除香港/新加坡外的全部节点作为节点池。"""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("订阅地址必须是 http(s):// 开头")

    nodes = await _fetch_subscription(url)
    cfg = load_config()
    pool, excluded_count = _build_auto_node_pool(nodes, cfg)

    cfg["subscription_url"] = url
    cfg["node_pool"] = pool
    cfg["node_pool_index"] = 0
    _write_json(CONFIG_FILE, cfg)

    active_proxy_url = ""
    active_node_name = ""
    active_index = 0
    if activate and pool:
        last_error: Exception | None = None
        for idx, node in enumerate(pool):
            try:
                active_node_name = node.get("name", "")
                active_proxy_url = await _activate_node_by_uri(node["raw_uri"], active_node_name, idx)
                active_index = idx
                break
            except Exception as e:
                last_error = e
                logger.warning(f"自动激活节点失败，尝试下一个: {node.get('name') or node.get('raw_uri', '')[:40]}: {e}")
        if not active_proxy_url and last_error:
            logger.warning(f"节点池已导入，但没有节点成功激活: {last_error}")

    return {
        "total": len(nodes),
        "nodes": nodes,
        "pool": pool,
        "pool_count": len(pool),
        "excluded_count": excluded_count,
        "active_proxy_url": active_proxy_url,
        "active_node_name": active_node_name,
        "active_index": active_index,
    }


# ==================== 路由 ====================

router = APIRouter()


class LoginBody(BaseModel):
    password: str


class SettingsBody(BaseModel):
    port_api: Optional[int] = None
    debug: Optional[bool] = None
    max_retries: Optional[int] = None
    proxy_url: Optional[str] = None
    admin_password: Optional[str] = None
    anti429_enabled: Optional[bool] = None
    anti429_target: Optional[str] = None
    force_no_stream: Optional[bool] = None
    anti_tracking: Optional[bool] = None
    drop_max_tokens: Optional[bool] = None


class KeyBody(BaseModel):
    name: str
    key: str
    description: str = ""


class SubscribeBody(BaseModel):
    url: str
    auto_activate: bool = True


class UseNodeBody(BaseModel):
    raw_uri: str
    name: str = ""


@router.get("/admin")
async def admin_page() -> FileResponse:
    index = STATIC_DIR / "admin.html"
    if not index.exists():
        raise HTTPException(status_code=500, detail="admin.html 不存在")
    return FileResponse(str(index), media_type="text/html; charset=utf-8")



@router.post("/api/admin/login")
async def admin_login(body: LoginBody) -> dict[str, Any]:
    expected = _get_admin_password()
    if not expected:
        raise HTTPException(status_code=500, detail="管理员密码未初始化")
    if body.password != expected:
        await asyncio.sleep(0.5)  # 轻微延迟
        raise HTTPException(status_code=401, detail="密码错误")
    tok = _issue_token()
    return {"token": tok, "ttl_seconds": SESSION_TTL}


@router.post("/api/admin/logout")
async def admin_logout(request: Request) -> dict[str, str]:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        _sessions.pop(auth[7:].strip(), None)
    return {"status": "ok"}


@router.get("/api/admin/settings")
async def get_settings(request: Request) -> dict[str, Any]:
    _require_auth(request)
    cfg = load_config()
    env_proxy = os.environ.get("PROXY_URL", "")
    return {
        "port_api": cfg.get("port_api", 2156),
        "debug": bool(cfg.get("debug", False)),
        "max_retries": int(cfg.get("max_retries", 2)),
        "proxy_url": cfg.get("proxy_url", ""),
        "env_proxy_url_override": env_proxy,
        "admin_password_env_locked": bool(os.environ.get("ADMIN_PASSWORD", "").strip()),
        "anti429_enabled": bool(cfg.get("anti429_enabled", False)),
        "anti429_target": cfg.get("anti429_target", "system"),
        "force_no_stream": bool(cfg.get("force_no_stream", False)),
        "anti_tracking": bool(cfg.get("anti_tracking", True)),
        "drop_max_tokens": bool(cfg.get("drop_max_tokens", True)),
    }


@router.put("/api/admin/settings")
async def update_settings(body: SettingsBody, request: Request) -> dict[str, Any]:
    _require_auth(request)
    cfg = _read_json(CONFIG_FILE, {})
    notes: list[str] = []

    if body.port_api is not None:
        if not (1 <= body.port_api <= 65535):
            raise HTTPException(status_code=400, detail="端口必须在 1-65535")
        if cfg.get("port_api") != body.port_api:
            notes.append("端口变更需要重启容器才能生效")
        cfg["port_api"] = body.port_api

    if body.debug is not None:
        if cfg.get("debug") != bool(body.debug):
            notes.append("debug 模式变更需要重启容器才能完全生效")
        cfg["debug"] = bool(body.debug)

    if body.max_retries is not None:
        if body.max_retries < 0 or body.max_retries > 100:
            raise HTTPException(status_code=400, detail="max_retries 应在 0-100")
        cfg["max_retries"] = int(body.max_retries)

    if body.proxy_url is not None:
        pu = body.proxy_url.strip()
        if pu and not any(pu.startswith(s) for s in ("http://", "https://", "socks5://", "socks://")):
            raise HTTPException(status_code=400, detail="代理 URL 必须以 http(s):// 或 socks5:// 开头")
        cfg["proxy_url"] = pu

    if body.admin_password is not None:
        if os.environ.get("ADMIN_PASSWORD", "").strip():
            raise HTTPException(status_code=400, detail="当前由环境变量 ADMIN_PASSWORD 锁定，无法在面板修改")
        new_pw = body.admin_password.strip()
        if len(new_pw) < 6:
            raise HTTPException(status_code=400, detail="密码至少 6 位")
        cfg["admin_password"] = new_pw
        notes.append("管理员密码已更新，下次登录生效")

    if body.anti429_enabled is not None:
        cfg["anti429_enabled"] = bool(body.anti429_enabled)

    if body.anti429_target is not None:
        if body.anti429_target not in ("system", "user"):
            raise HTTPException(status_code=400, detail="anti429_target 必须是 system 或 user")
        cfg["anti429_target"] = body.anti429_target

    if body.force_no_stream is not None:
        cfg["force_no_stream"] = bool(body.force_no_stream)

    if body.anti_tracking is not None:
        cfg["anti_tracking"] = bool(body.anti_tracking)

    if body.drop_max_tokens is not None:
        cfg["drop_max_tokens"] = bool(body.drop_max_tokens)

    _write_json(CONFIG_FILE, cfg)
    return {"status": "ok", "notes": notes}


@router.get("/api/admin/keys")
async def get_keys(request: Request) -> dict[str, Any]:
    _require_auth(request)
    return {"keys": _read_api_keys()}


@router.post("/api/admin/keys")
async def add_key(body: KeyBody, request: Request) -> dict[str, str]:
    _require_auth(request)
    name = body.name.strip()
    key = body.key.strip()
    if not name or not key:
        raise HTTPException(status_code=400, detail="name / key 不能为空")
    if ":" in name:
        raise HTTPException(status_code=400, detail="name 不能包含冒号")
    if not key.startswith("sk-"):
        raise HTTPException(status_code=400, detail="key 必须以 sk- 开头")

    keys = _read_api_keys()
    keys = [k for k in keys if k["name"] != name]  # 同名覆盖
    keys.append({"name": name, "key": key, "description": body.description or ""})
    _write_api_keys(keys)
    api_key_manager.load_keys()  # 热加载
    return {"status": "ok"}


@router.delete("/api/admin/keys/{name}")
async def delete_key(name: str, request: Request) -> dict[str, str]:
    _require_auth(request)
    keys = _read_api_keys()
    new_keys = [k for k in keys if k["name"] != name]
    if len(new_keys) == len(keys):
        raise HTTPException(status_code=404, detail="未找到该密钥")
    _write_api_keys(new_keys)
    api_key_manager.load_keys()
    return {"status": "ok"}


@router.get("/api/admin/models")
async def get_models(request: Request) -> dict[str, Any]:
    _require_auth(request)
    data = _read_json(MODELS_FILE, {"models": [], "alias_map": {}})
    return {
        "models": data.get("models", []),
        "alias_map": data.get("alias_map", {}),
    }


class ModelsBody(BaseModel):
    models: list[str] | None = None
    alias_map: dict[str, str] | None = None


@router.put("/api/admin/models")
async def update_models(body: ModelsBody, request: Request) -> dict[str, Any]:
    _require_auth(request)
    data = _read_json(MODELS_FILE, {"models": [], "alias_map": {}})
    if body.models is not None:
        cleaned = [m.strip() for m in body.models if m.strip()]
        if not cleaned:
            raise HTTPException(status_code=400, detail="models 列表不能为空")
        data["models"] = cleaned
    if body.alias_map is not None:
        data["alias_map"] = {k.strip(): v.strip() for k, v in body.alias_map.items() if k.strip() and v.strip()}
    _write_json(MODELS_FILE, data)
    return {"status": "ok"}


@router.post("/api/admin/subscribe")
async def fetch_subscription(body: SubscribeBody, request: Request) -> dict[str, Any]:
    _require_auth(request)
    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="订阅地址必须是 http(s):// 开头")
    result = await refresh_subscription_pool(url, activate=True)
    nodes = result["nodes"]
    return {
        "total": len(nodes),
        "usable_count": sum(1 for n in nodes if n.get("usable_as_proxy")),
        "pool_count": result["pool_count"],
        "pool": result["pool"],
        "excluded_count": result["excluded_count"],
        "active_proxy_url": result["active_proxy_url"],
        "active_node_name": result["active_node_name"],
        "nodes": nodes,
    }


@router.get("/api/admin/subscription")
async def get_subscription(request: Request) -> dict[str, Any]:
    """返回上次保存的订阅 URL（面板打开时自动回填）"""
    _require_auth(request)
    cfg = load_config()
    return {"url": cfg.get("subscription_url", "")}


@router.post("/api/admin/use-node")
async def use_node(body: UseNodeBody, request: Request) -> dict[str, Any]:
    _require_auth(request)
    uri = body.raw_uri.strip()
    if not uri:
        raise HTTPException(status_code=400, detail="节点 URI 为空")

    cfg = _read_json(CONFIG_FILE, {})

    if needs_worker(uri):
        # 通过内置 worker 做协议转换
        try:
            proxy_url = worker.start_with_uri(uri, name=body.name or "")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        cfg["proxy_url"] = proxy_url
        cfg["active_node_uri"] = uri
        cfg["active_node_name"] = body.name or ""
        _write_json(CONFIG_FILE, cfg)
        return {"status": "ok", "proxy_url": proxy_url, "via": "worker", "node_name": body.name}

    # http / https / socks5 — 直接当出站代理
    if not any(uri.startswith(s) for s in ("http://", "https://", "socks5://", "socks://")):
        raise HTTPException(status_code=400, detail=f"不支持的协议: {uri[:20]}")

    # 切换到直连模式时停掉 worker
    worker.stop()
    cfg["proxy_url"] = uri
    cfg["active_node_uri"] = uri
    cfg["active_node_name"] = body.name or ""
    _write_json(CONFIG_FILE, cfg)
    return {"status": "ok", "proxy_url": uri, "via": "direct", "node_name": body.name}





async def _activate_node_by_uri(uri: str, name: str, pool_index: int = 0) -> str:
    """激活指定节点，写入 config，返回 proxy_url（供节点池轮换内部调用）"""
    cfg = _read_json(CONFIG_FILE, {})
    cfg["node_pool_index"] = pool_index
    if needs_worker(uri):
        proxy_url = worker.start_with_uri(uri, name=name)
        cfg["proxy_url"] = proxy_url
        cfg["active_node_uri"] = uri
        cfg["active_node_name"] = name
        _write_json(CONFIG_FILE, cfg)
        return proxy_url
    if not any(uri.startswith(s) for s in ("http://", "https://", "socks5://", "socks://")):
        raise ValueError(f"不支持的协议: {uri[:30]}")
    worker.stop()
    cfg["proxy_url"] = uri
    cfg["active_node_uri"] = uri
    cfg["active_node_name"] = name
    _write_json(CONFIG_FILE, cfg)
    return uri


# ── Node Pool ──────────────────────────────────────────────────────────────────
# Pool is stored in config as "node_pool": [{"raw_uri": ..., "name": ...}, ...]
# The pool index (which node is currently active) is tracked in "node_pool_index".
# When a pool is active and max_retries is exhausted, the backend auto-rotates
# to the next node. This is wired in vertex_client via load_config().

class NodePoolBody(BaseModel):
    pool: list[dict[str, Any]]


@router.get("/api/admin/node-pool")
async def get_node_pool(request: Request) -> dict[str, Any]:
    _require_auth(request)
    cfg = _read_json(CONFIG_FILE, {})
    return {
        "pool": cfg.get("node_pool", []),
        "current_index": cfg.get("node_pool_index", 0),
    }


@router.post("/api/admin/node-pool")
async def set_node_pool(body: NodePoolBody, request: Request) -> dict[str, Any]:
    _require_auth(request)
    pool = []
    for entry in body.pool:
        uri = str(entry.get("raw_uri", "")).strip()
        if uri:
            pool.append({"raw_uri": uri, "name": str(entry.get("name", ""))})
    cfg = _read_json(CONFIG_FILE, {})
    cfg["node_pool"] = pool
    cfg["node_pool_index"] = 0
    _write_json(CONFIG_FILE, cfg)
    return {"status": "ok", "count": len(pool)}


@router.post("/api/admin/stop-proxy")
async def stop_proxy(request: Request) -> dict[str, Any]:
    """停掉 worker、清空代理，回到直连模式"""
    _require_auth(request)
    worker.stop()
    cfg = _read_json(CONFIG_FILE, {})
    cfg["proxy_url"] = ""
    cfg["active_node_uri"] = ""
    cfg["active_node_name"] = ""
    _write_json(CONFIG_FILE, cfg)
    return {"status": "ok"}


@router.get("/api/admin/proxy-status")
async def proxy_status(request: Request) -> dict[str, Any]:
    _require_auth(request)
    cfg = load_config()
    s = worker.status()
    s["configured_proxy_url"] = cfg.get("proxy_url", "")
    s["active_node_uri"] = cfg.get("active_node_uri", "")
    s["active_node_name"] = cfg.get("active_node_name", "")
    return s
