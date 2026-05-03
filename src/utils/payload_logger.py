"""
payload_logger.py

当 config.log_payload=true 时，将每次请求的原始 fetch body 和原始 response
分别保存到 logs/fetch_N.txt 和 logs/response_N.txt（N=1/2/3，循环覆盖）。
"""

import json
import os
from pathlib import Path

from src.core.config import load_config

_MAX_FILES = 3
_counter = {"fetch": 0, "response": 0}


def _log_dir() -> Path:
    config = load_config()
    d = Path(config.get("log_dir", "logs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_enabled() -> bool:
    return bool(load_config().get("log_payload", False))


def _next_slot(kind: str) -> int:
    """返回下一个槽位编号（1~3 循环）"""
    _counter[kind] = (_counter[kind] % _MAX_FILES) + 1
    return _counter[kind]


def save_fetch(body: dict) -> None:
    """保存发送给 Vertex 的原始请求 body（dict -> JSON 字符串）"""
    if not _is_enabled():
        return
    slot = _next_slot("fetch")
    path = _log_dir() / f"fetch-{slot:02d}.txt"
    try:
        path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def save_response(raw_chunks: list) -> None:
    """保存从 Vertex 收到的原始响应行列表，不做任何处理"""
    if not _is_enabled():
        return
    slot = _next_slot("response")
    path = _log_dir() / f"response-{slot:02d}.txt"
    try:
        # raw_chunks 是逐行读出的字符串列表，原样拼接还原
        path.write_text("".join(raw_chunks), encoding="utf-8")
    except Exception:
        pass
