"""代理配置解析工具。"""

import importlib.util
from typing import Any, Dict, List, Optional


VALID_PROXY_SCHEMES = {"socks5", "http", "auto"}


def get_proxy_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    从配置中读取代理设置。

    代理配置字段：
    - enabled: 是否启用代理
    - scheme: socks5/http/auto
    - host: 代理主机
    - port: 代理端口
    """
    proxy = config.get("proxy", {}) if isinstance(config, dict) else {}

    enabled = bool(proxy.get("enabled", False))
    scheme = str(proxy.get("scheme", "socks5")).strip().lower()
    host = str(proxy.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port_raw = proxy.get("port", 7890)

    if scheme not in VALID_PROXY_SCHEMES:
        scheme = "socks5"

    try:
        port = int(port_raw)
    except (TypeError, ValueError):
        port = 7890

    if port <= 0 or port > 65535:
        port = 7890

    return {
        "enabled": enabled,
        "scheme": scheme,
        "host": host,
        "port": port
    }


def build_proxy_candidates(config: Dict[str, Any]) -> List[str]:
    """根据配置生成可用代理 URL 列表。"""
    settings = get_proxy_settings(config)
    if not settings["enabled"]:
        return []

    host = settings["host"]
    port = settings["port"]
    scheme = settings["scheme"]

    # 读取可选的认证信息
    proxy_block = config.get("proxy", {}) if isinstance(config, dict) else {}
    username = str(proxy_block.get("username", "")).strip()
    password = str(proxy_block.get("password", "")).strip()
    auth = f"{username}:{password}@" if username and password else ""

    if scheme == "auto":
        # 如果未安装 socks 支持，优先尝试 HTTP 代理，避免每次请求先失败一次。
        socks_supported = importlib.util.find_spec("socksio") is not None
        schemes = ["socks5", "http"] if socks_supported else ["http", "socks5"]
    else:
        schemes = [scheme]
    candidates = [f"{s}://{auth}{host}:{port}" for s in schemes]

    # 去重并保持顺序
    unique: List[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def get_primary_proxy(config: Dict[str, Any]) -> Optional[str]:
    """返回首选代理 URL（若未启用则返回 None）。"""
    candidates = build_proxy_candidates(config)
    return candidates[0] if candidates else None
