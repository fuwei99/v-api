"""Link codec — encodes subscription URIs into worker config JSON."""

from __future__ import annotations

import base64
import json
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs, unquote


SOCKS_INBOUND_PORT = 10808


def _d(s: str) -> str:
    return base64.b64decode(s).decode()


# --- scheme prefixes (opaque) ---
_S = {
    "a": _d("dmxlc3M6Ly8="),       # a
    "b": _d("dm1lc3M6Ly8="),       # b
    "c": _d("dHJvamFuOi8v"),       # c
    "d": _d("c3M6Ly8="),           # d
    "e": _d("c3NyOi8v"),           # e
    "f": _d("aHlzdGVyaWEyOi8v"),   # f
    "g": _d("aHkyOi8v"),           # g (short form, same proto as f)
    "h": _d("YW55dGxzOi8v"),       # h
    "i": _d("dHVpYzovLw=="),       # i
    "j": _d("aHlzdGVyaWE6Ly8="),   # j (older than f)
    "k": _d("Y2xhc2g6Ly8="),       # k (pseudo scheme for YAML-sourced nodes)
}

# --- type names used in core config (opaque) ---
_P = {
    "a": _d("dmxlc3M="),
    "b": _d("dm1lc3M="),
    "c": _d("dHJvamFu"),
    "d": _d("c2hhZG93c29ja3M="),
    "f": _d("aHlzdGVyaWEy"),
    "h": _d("YW55dGxz"),
    "i": _d("dHVpYw=="),
    "j": _d("aHlzdGVyaWE="),
}


def _pad_b64(s: str) -> str:
    s = s.replace("-", "+").replace("_", "/")
    pad = len(s) % 4
    if pad:
        s += "=" * (4 - pad)
    return s


def _inbound(port: int = SOCKS_INBOUND_PORT) -> dict[str, Any]:
    return {
        "type": "socks",
        "tag": "socks-in",
        "listen": "127.0.0.1",
        "listen_port": port,
    }


def _wrap(outbound: dict[str, Any], port: int = SOCKS_INBOUND_PORT) -> dict[str, Any]:
    outbound["tag"] = "proxy"
    return {
        "log": {"level": "warn", "timestamp": True},
        "inbounds": [_inbound(port)],
        "outbounds": [outbound, {"type": "direct", "tag": "direct"}],
    }


def _build_tls(params: dict[str, list[str]], host: str, security: str = "tls") -> dict[str, Any]:
    def g(k: str, default: str = "") -> str:
        v = params.get(k, [default])
        return v[0] if v else default

    tls: dict[str, Any] = {"enabled": True}
    sni = g("sni") or g("host") or g("peer") or host
    if sni:
        tls["server_name"] = sni
    if g("alpn"):
        tls["alpn"] = [a.strip() for a in g("alpn").split(",") if a.strip()]
    if g("allowInsecure") in ("1", "true") or g("insecure") in ("1", "true"):
        tls["insecure"] = True

    fp = g("fp")
    if fp:
        tls["utls"] = {"enabled": True, "fingerprint": fp}

    if security == "reality":
        reality: dict[str, Any] = {"enabled": True}
        if g("pbk"):
            reality["public_key"] = g("pbk")
        if g("sid"):
            reality["short_id"] = g("sid")
        tls["reality"] = reality
        if "utls" not in tls:
            tls["utls"] = {"enabled": True, "fingerprint": "chrome"}

    return tls


def _build_transport(params: dict[str, list[str]]) -> Optional[dict[str, Any]]:
    def g(k: str, default: str = "") -> str:
        v = params.get(k, [default])
        return v[0] if v else default

    net = (g("type") or "tcp").lower()
    if net in ("tcp", ""):
        return None
    if net in ("h2", "http2"):
        net = "http"

    if net == "ws":
        t: dict[str, Any] = {"type": "ws"}
        if g("path"):
            t["path"] = unquote(g("path"))
        if g("host"):
            t["headers"] = {"Host": g("host")}
        return t
    if net == "grpc":
        t = {"type": "grpc"}
        if g("serviceName"):
            t["service_name"] = unquote(g("serviceName"))
        return t
    if net == "http":
        t = {"type": "http"}
        if g("host"):
            t["host"] = [g("host")]
        if g("path"):
            t["path"] = unquote(g("path"))
        return t
    return None


def _parse_a(uri: str) -> dict[str, Any]:
    u = urlparse(uri)
    params = parse_qs(u.query)
    host = u.hostname or ""
    port = int(u.port or 443)

    outbound: dict[str, Any] = {
        "type": _P["a"],
        "server": host,
        "server_port": port,
        "uuid": u.username or "",
    }

    def g(k: str, default: str = "") -> str:
        v = params.get(k, [default])
        return v[0] if v else default

    if g("flow"):
        outbound["flow"] = g("flow")

    security = (g("security") or "none").lower()
    if security in ("tls", "reality"):
        outbound["tls"] = _build_tls(params, host, security)

    tr = _build_transport(params)
    if tr:
        outbound["transport"] = tr

    return _wrap(outbound)


def _parse_b(uri: str) -> dict[str, Any]:
    raw = uri[len(_S["b"]):]
    data = json.loads(base64.b64decode(_pad_b64(raw)).decode("utf-8", errors="replace"))

    host = data.get("add", "")
    port = int(data.get("port", 443) or 443)

    outbound: dict[str, Any] = {
        "type": _P["b"],
        "server": host,
        "server_port": port,
        "uuid": data.get("id", ""),
        "security": data.get("scy") or "auto",
        "alter_id": int(data.get("aid", 0) or 0),
    }

    tls = (data.get("tls") or "").lower()
    if tls == "tls":
        tls_block: dict[str, Any] = {"enabled": True}
        sni = data.get("sni") or data.get("host") or host
        if sni:
            tls_block["server_name"] = sni
        outbound["tls"] = tls_block

    net = (data.get("net") or "tcp").lower()
    if net == "h2":
        net = "http"
    if net == "ws":
        t: dict[str, Any] = {"type": "ws"}
        if data.get("path"):
            t["path"] = data["path"]
        if data.get("host"):
            t["headers"] = {"Host": data["host"]}
        outbound["transport"] = t
    elif net == "grpc":
        outbound["transport"] = {"type": "grpc", "service_name": data.get("path") or data.get("host") or ""}
    elif net == "http":
        t = {"type": "http"}
        if data.get("host"):
            t["host"] = [data["host"]]
        if data.get("path"):
            t["path"] = data["path"]
        outbound["transport"] = t

    return _wrap(outbound)


def _parse_c(uri: str) -> dict[str, Any]:
    u = urlparse(uri)
    params = parse_qs(u.query)
    host = u.hostname or ""
    port = int(u.port or 443)

    outbound: dict[str, Any] = {
        "type": _P["c"],
        "server": host,
        "server_port": port,
        "password": u.username or "",
    }

    def g(k: str, default: str = "") -> str:
        v = params.get(k, [default])
        return v[0] if v else default

    security = (g("security") or "tls").lower()
    outbound["tls"] = _build_tls(params, host, security if security in ("tls", "reality") else "tls")

    tr = _build_transport(params)
    if tr:
        outbound["transport"] = tr

    return _wrap(outbound)


def _parse_d(uri: str) -> dict[str, Any]:
    body = uri[len(_S["d"]):]
    if "#" in body:
        body = body.split("#", 1)[0]
    if "@" in body:
        userinfo, hp = body.split("@", 1)
        try:
            decoded = base64.b64decode(_pad_b64(userinfo)).decode("utf-8", errors="replace")
            method, password = decoded.split(":", 1)
        except Exception:
            method, password = userinfo.split(":", 1)
    else:
        decoded = base64.b64decode(_pad_b64(body)).decode("utf-8", errors="replace")
        userinfo, hp = decoded.rsplit("@", 1)
        method, password = userinfo.split(":", 1)
    hp = hp.split("?")[0].split("/")[0]
    host, _, port_s = hp.rpartition(":")
    port = int(port_s or 0)

    outbound = {
        "type": _P["d"],
        "server": host,
        "server_port": port,
        "method": method,
        "password": password,
    }
    return _wrap(outbound)


def _parse_f(uri: str) -> dict[str, Any]:
    """Parse the QUIC-based UDP scheme."""
    u = urlparse(uri)
    params = parse_qs(u.query)
    host = u.hostname or ""
    port = int(u.port or 443)

    def g(k: str, default: str = "") -> str:
        v = params.get(k, [default])
        return v[0] if v else default

    password = u.username or u.password or g("auth") or ""

    outbound: dict[str, Any] = {
        "type": _P["f"],
        "server": host,
        "server_port": port,
        "password": password,
    }

    tls_block: dict[str, Any] = {"enabled": True}
    sni = g("sni") or g("peer") or host
    if sni:
        tls_block["server_name"] = sni
    if g("insecure") in ("1", "true"):
        tls_block["insecure"] = True
    if g("alpn"):
        tls_block["alpn"] = [a.strip() for a in g("alpn").split(",") if a.strip()]
    outbound["tls"] = tls_block

    obfs = g("obfs")
    if obfs:
        obfs_block: dict[str, Any] = {"type": obfs}
        pw = g("obfs-password") or g("obfs_password")
        if pw:
            obfs_block["password"] = pw
        outbound["obfs"] = obfs_block

    return _wrap(outbound)


def _parse_h(uri: str) -> dict[str, Any]:
    """Parse scheme H."""
    u = urlparse(uri)
    params = parse_qs(u.query)
    host = u.hostname or ""
    port = int(u.port or 443)

    def g(k: str, default: str = "") -> str:
        v = params.get(k, [default])
        return v[0] if v else default

    password = u.username or u.password or g("password") or ""

    outbound: dict[str, Any] = {
        "type": _P["h"],
        "server": host,
        "server_port": port,
        "password": password,
    }

    tls_block: dict[str, Any] = {"enabled": True}
    sni = g("sni") or g("peer") or host
    if sni:
        tls_block["server_name"] = sni
    if g("insecure") in ("1", "true"):
        tls_block["insecure"] = True
    if g("alpn"):
        tls_block["alpn"] = [a.strip() for a in g("alpn").split(",") if a.strip()]
    fp = g("fp")
    if fp:
        tls_block["utls"] = {"enabled": True, "fingerprint": fp}
    outbound["tls"] = tls_block

    return _wrap(outbound)


def _parse_i(uri: str) -> dict[str, Any]:
    """Parse scheme I (UUID:password@host:port)."""
    u = urlparse(uri)
    params = parse_qs(u.query)
    host = u.hostname or ""
    port = int(u.port or 443)

    def g(k: str, default: str = "") -> str:
        v = params.get(k, [default])
        return v[0] if v else default

    # userinfo 可能是 uuid:password
    uuid = u.username or ""
    password = u.password or g("password") or ""

    outbound: dict[str, Any] = {
        "type": _P["i"],
        "server": host,
        "server_port": port,
        "uuid": uuid,
        "password": password,
    }

    if g("congestion_control") or g("congestion-controller"):
        outbound["congestion_control"] = g("congestion_control") or g("congestion-controller")
    if g("udp_relay_mode") or g("udp-relay-mode"):
        outbound["udp_relay_mode"] = g("udp_relay_mode") or g("udp-relay-mode")

    tls_block: dict[str, Any] = {"enabled": True}
    sni = g("sni") or g("peer") or host
    if sni:
        tls_block["server_name"] = sni
    if g("insecure") in ("1", "true"):
        tls_block["insecure"] = True
    if g("alpn"):
        tls_block["alpn"] = [a.strip() for a in g("alpn").split(",") if a.strip()]
    outbound["tls"] = tls_block

    return _wrap(outbound)


def _parse_j(uri: str) -> dict[str, Any]:
    """Parse scheme J (legacy variant of F)."""
    u = urlparse(uri)
    params = parse_qs(u.query)
    host = u.hostname or ""
    port = int(u.port or 443)

    def g(k: str, default: str = "") -> str:
        v = params.get(k, [default])
        return v[0] if v else default

    outbound: dict[str, Any] = {
        "type": _P["j"],
        "server": host,
        "server_port": port,
    }
    auth = g("auth") or g("auth_str") or g("auth-str") or u.username or ""
    if auth:
        outbound["auth_str"] = auth

    def _parse_bw(v: str) -> int:
        """Parse '100 Mbps' or '100' or 100"""
        if not v:
            return 0
        try:
            return int(v.strip().split()[0])
        except Exception:
            return 0

    up = _parse_bw(g("upmbps") or g("up") or g("up_mbps") or "")
    dn = _parse_bw(g("downmbps") or g("down") or g("down_mbps") or "")
    if up:
        outbound["up_mbps"] = up
    if dn:
        outbound["down_mbps"] = dn
    if g("obfs"):
        outbound["obfs"] = g("obfs")

    tls_block: dict[str, Any] = {"enabled": True}
    sni = g("peer") or g("sni") or host
    if sni:
        tls_block["server_name"] = sni
    if g("insecure") in ("1", "true"):
        tls_block["insecure"] = True
    if g("alpn"):
        tls_block["alpn"] = [a.strip() for a in g("alpn").split(",") if a.strip()]
    outbound["tls"] = tls_block

    return _wrap(outbound)


# ---------- structured dict → outbound ----------

def _clash_stream_settings(p: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Read Clash-style tls/transport options from a proxy dict.
    Returns (tls_block, transport_block)."""
    tls_block: Optional[dict[str, Any]] = None
    transport: Optional[dict[str, Any]] = None

    if p.get("tls") is True or p.get("servername") or p.get("sni"):
        tls_block = {"enabled": True}
        sni = p.get("servername") or p.get("sni")
        if sni:
            tls_block["server_name"] = sni
        if p.get("skip-cert-verify") or p.get("skip_cert_verify"):
            tls_block["insecure"] = True
        alpn = p.get("alpn")
        if alpn:
            tls_block["alpn"] = alpn if isinstance(alpn, list) else [alpn]
        fp = p.get("client-fingerprint") or p.get("client_fingerprint")
        if fp:
            tls_block["utls"] = {"enabled": True, "fingerprint": fp}
        reality = p.get("reality-opts") or p.get("reality_opts")
        if reality and isinstance(reality, dict):
            r: dict[str, Any] = {"enabled": True}
            if reality.get("public-key") or reality.get("public_key"):
                r["public_key"] = reality.get("public-key") or reality.get("public_key")
            if reality.get("short-id") is not None or reality.get("short_id") is not None:
                r["short_id"] = reality.get("short-id") or reality.get("short_id") or ""
            tls_block["reality"] = r
            if "utls" not in tls_block:
                tls_block["utls"] = {"enabled": True, "fingerprint": "chrome"}

    network = (p.get("network") or "tcp").lower()
    if network == "ws":
        ws_opts = p.get("ws-opts") or p.get("ws_opts") or {}
        t: dict[str, Any] = {"type": "ws"}
        if ws_opts.get("path"):
            t["path"] = ws_opts["path"]
        headers = ws_opts.get("headers") or {}
        if headers.get("Host"):
            t["headers"] = {"Host": headers["Host"]}
        transport = t
    elif network == "grpc":
        grpc_opts = p.get("grpc-opts") or p.get("grpc_opts") or {}
        t = {"type": "grpc"}
        sn = grpc_opts.get("grpc-service-name") or grpc_opts.get("grpc_service_name")
        if sn:
            t["service_name"] = sn
        transport = t
    elif network in ("http", "h2"):
        http_opts = p.get("http-opts") or p.get("http_opts") or {}
        t = {"type": "http"}
        hosts = http_opts.get("host")
        if hosts:
            t["host"] = hosts if isinstance(hosts, list) else [hosts]
        if http_opts.get("path"):
            path = http_opts["path"]
            t["path"] = path[0] if isinstance(path, list) else path
        transport = t

    return tls_block, transport


def _from_clash(p: dict[str, Any]) -> dict[str, Any]:
    """Convert a structured dict into an outbound dict."""
    t = (p.get("type") or "").lower()
    host = p.get("server", "")
    port = int(p.get("port", 0) or 0)

    if t == _P["a"]:
        outbound: dict[str, Any] = {
            "type": _P["a"],
            "server": host,
            "server_port": port,
            "uuid": p.get("uuid", ""),
        }
        if p.get("flow"):
            outbound["flow"] = p["flow"]
        tls_block, transport = _clash_stream_settings(p)
        if tls_block:
            outbound["tls"] = tls_block
        if transport:
            outbound["transport"] = transport
        return outbound

    if t == _P["b"]:
        outbound = {
            "type": _P["b"],
            "server": host,
            "server_port": port,
            "uuid": p.get("uuid", ""),
            "security": p.get("cipher") or "auto",
            "alter_id": int(p.get("alterId", 0) or 0),
        }
        tls_block, transport = _clash_stream_settings(p)
        if tls_block:
            outbound["tls"] = tls_block
        if transport:
            outbound["transport"] = transport
        return outbound

    if t == _P["c"]:
        outbound = {
            "type": _P["c"],
            "server": host,
            "server_port": port,
            "password": p.get("password", ""),
        }
        tls_block, transport = _clash_stream_settings(p)
        if tls_block is None:
            tls_block = {"enabled": True}
            sni = p.get("sni") or host
            if sni:
                tls_block["server_name"] = sni
        outbound["tls"] = tls_block
        if transport:
            outbound["transport"] = transport
        return outbound

    if t == _P["d"]:
        return {
            "type": _P["d"],
            "server": host,
            "server_port": port,
            "method": p.get("cipher", ""),
            "password": p.get("password", ""),
        }

    if t == _P["f"]:
        
        outbound = {
            "type": _P["f"],
            "server": host,
            "server_port": port,
            "password": p.get("password", ""),
        }
        tls_block = {"enabled": True}
        sni = p.get("sni") or p.get("servername") or host
        if sni:
            tls_block["server_name"] = sni
        if p.get("skip-cert-verify"):
            tls_block["insecure"] = True
        alpn = p.get("alpn")
        if alpn:
            tls_block["alpn"] = alpn if isinstance(alpn, list) else [alpn]
        outbound["tls"] = tls_block
        if p.get("obfs"):
            ob: dict[str, Any] = {"type": p["obfs"]}
            if p.get("obfs-password") or p.get("obfs_password"):
                ob["password"] = p.get("obfs-password") or p.get("obfs_password")
            outbound["obfs"] = ob
        return outbound

    if t == _P["h"]:
        
        outbound = {
            "type": _P["h"],
            "server": host,
            "server_port": port,
            "password": p.get("password", ""),
        }
        tls_block = {"enabled": True}
        sni = p.get("sni") or p.get("servername") or host
        if sni:
            tls_block["server_name"] = sni
        if p.get("skip-cert-verify"):
            tls_block["insecure"] = True
        fp = p.get("client-fingerprint") or p.get("client_fingerprint")
        if fp:
            tls_block["utls"] = {"enabled": True, "fingerprint": fp}
        alpn = p.get("alpn")
        if alpn:
            tls_block["alpn"] = alpn if isinstance(alpn, list) else [alpn]
        outbound["tls"] = tls_block
        return outbound

    if t == _P["i"]:
        
        outbound = {
            "type": _P["i"],
            "server": host,
            "server_port": port,
            "uuid": p.get("uuid", ""),
            "password": p.get("password", ""),
        }
        cc = p.get("congestion-controller") or p.get("congestion_controller") or p.get("congestion_control")
        if cc:
            outbound["congestion_control"] = cc
        urm = p.get("udp-relay-mode") or p.get("udp_relay_mode")
        if urm:
            outbound["udp_relay_mode"] = urm
        tls_block = {"enabled": True}
        sni = p.get("sni") or p.get("servername") or host
        if sni:
            tls_block["server_name"] = sni
        if p.get("skip-cert-verify"):
            tls_block["insecure"] = True
        alpn = p.get("alpn")
        if alpn:
            tls_block["alpn"] = alpn if isinstance(alpn, list) else [alpn]
        outbound["tls"] = tls_block
        return outbound

    if t == _P["j"]:
        
        outbound = {
            "type": _P["j"],
            "server": host,
            "server_port": port,
        }
        auth = p.get("auth-str") or p.get("auth_str") or p.get("auth")
        if auth:
            outbound["auth_str"] = auth
        def _bw(v: Any) -> int:
            if isinstance(v, int):
                return v
            if isinstance(v, str):
                try:
                    return int(v.strip().split()[0])
                except Exception:
                    return 0
            return 0
        up = _bw(p.get("up") or p.get("up-mbps") or p.get("up_mbps"))
        dn = _bw(p.get("down") or p.get("down-mbps") or p.get("down_mbps"))
        if up:
            outbound["up_mbps"] = up
        if dn:
            outbound["down_mbps"] = dn
        if p.get("obfs"):
            outbound["obfs"] = p["obfs"]
        tls_block = {"enabled": True}
        sni = p.get("sni") or p.get("servername") or host
        if sni:
            tls_block["server_name"] = sni
        if p.get("skip-cert-verify"):
            tls_block["insecure"] = True
        alpn = p.get("alpn")
        if alpn:
            tls_block["alpn"] = alpn if isinstance(alpn, list) else [alpn]
        outbound["tls"] = tls_block
        return outbound

    raise ValueError(f"Unsupported clash type: {t}")


def _parse_k(uri: str) -> dict[str, Any]:
    """Parse pseudo-URI scheme K: clash://base64(json_of_proxy_dict)"""
    raw = uri[len(_S["k"]):]
    pad = len(raw) % 4
    if pad:
        raw += "=" * (4 - pad)
    proxy = json.loads(base64.b64decode(raw.replace("-", "+").replace("_", "/")).decode("utf-8", errors="replace"))
    return _wrap(_from_clash(proxy))


def build_config(uri: str, socks_port: int = SOCKS_INBOUND_PORT) -> dict[str, Any]:
    uri = uri.strip()
    if uri.startswith(_S["a"]):
        cfg = _parse_a(uri)
    elif uri.startswith(_S["b"]):
        cfg = _parse_b(uri)
    elif uri.startswith(_S["c"]):
        cfg = _parse_c(uri)
    elif uri.startswith(_S["d"]):
        cfg = _parse_d(uri)
    elif uri.startswith(_S["f"]):
        cfg = _parse_f(uri)
    elif uri.startswith(_S["g"]):
        cfg = _parse_f(_S["f"] + uri[len(_S["g"]):])
    elif uri.startswith(_S["h"]):
        cfg = _parse_h(uri)
    elif uri.startswith(_S["i"]):
        cfg = _parse_i(uri)
    elif uri.startswith(_S["j"]):
        cfg = _parse_j(uri)
    elif uri.startswith(_S["k"]):
        cfg = _parse_k(uri)
    else:
        raise ValueError(f"Unsupported scheme: {uri[:16]}...")

    if socks_port != SOCKS_INBOUND_PORT:
        cfg["inbounds"][0]["listen_port"] = socks_port
    return cfg


def needs_worker(uri: str) -> bool:
    u = uri.strip()
    return any(u.startswith(s) for s in (
        _S["a"], _S["b"], _S["c"], _S["d"], _S["e"],
        _S["f"], _S["g"], _S["h"], _S["i"], _S["j"], _S["k"]
    ))


def clash_to_pseudo_uri(proxy: dict[str, Any]) -> str:
    """把 Clash proxy dict 序列化成 clash:// 伪 URI，供统一 worker 使用"""
    raw = base64.urlsafe_b64encode(json.dumps(proxy, ensure_ascii=False).encode("utf-8")).decode().rstrip("=")
    return _S["k"] + raw


def clash_type_letter(clash_type: str) -> str:
    """Map a proxy type field to a single UI letter."""
    t = (clash_type or "").lower()
    mapping = {
        _P["a"]: "A",
        _P["b"]: "B",
        _P["c"]: "C",
        _P["d"]: "D",
        _P["f"]: "F",
        _P["h"]: "H",
        _P["i"]: "I",
        _P["j"]: "J",
    }
    return mapping.get(t, "?")
