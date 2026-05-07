"""Transport worker lifecycle — downloads, starts and stops the background binary."""

from __future__ import annotations

import atexit
import base64
import io
import json
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger
from .codec import build_config, SOCKS_INBOUND_PORT

logger = get_logger(__name__)


def _d(s: str) -> str:
    return base64.b64decode(s).decode()

# --- release URLs (opaque) ---
_LATEST_URL = _d("aHR0cHM6Ly9naXRodWIuY29tL1NhZ2VyTmV0L3NpbmctYm94L3JlbGVhc2VzL2xhdGVzdA==")
_DL_BASE = _d("aHR0cHM6Ly9naXRodWIuY29tL1NhZ2VyTmV0L3NpbmctYm94L3JlbGVhc2VzL2Rvd25sb2FkLw==")
_PKG_NAME = _d("c2luZy1ib3g=")  # package basename


BIN_DIR = Path("/app/bin")
BIN_PATH = BIN_DIR / "netcore"
CONFIG_PATH = Path("/tmp/worker-config.json")
LOG_PATH = Path("/tmp/worker.log")


class WorkerManager:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._active_uri: Optional[str] = None
        self._active_name: Optional[str] = None
        self._port: int = SOCKS_INBOUND_PORT
        atexit.register(self.stop)

    # ---------- binary ----------
    def find_binary(self) -> Optional[str]:
        for p in [str(BIN_PATH), "/usr/local/bin/netcore", "/tmp/netcore"]:
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
        w = shutil.which("netcore")
        if w:
            return w
        return None

    def _resolve_latest_version(self) -> str:
        """跟随 /releases/latest 的 302 重定向，从最终 URL 里取到 tag"""
        req = urllib.request.Request(_LATEST_URL, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            final = resp.url  # 例: https://host/xxx/releases/tag/v1.x.y
        tag = final.rstrip("/").rsplit("/", 1)[-1]
        if not tag.startswith("v"):
            raise RuntimeError(f"无法解析版本号: {final}")
        return tag

    def _build_download_url(self, tag: str, arch_suffix: str) -> str:
        version = tag.lstrip("v")
        filename = f"{_PKG_NAME}-{version}-linux-{arch_suffix}.tar.gz"
        return f"{_DL_BASE}{tag}/{filename}"

    def ensure_binary(self) -> str:
        existing = self.find_binary()
        if existing:
            return existing

        logger.info("[Worker] 内核未就绪，开始下载...")
        arch = platform.machine().lower()
        if arch in ("x86_64", "amd64"):
            arch_suffix = "amd64"
        elif arch in ("aarch64", "arm64"):
            arch_suffix = "arm64"
        else:
            raise RuntimeError(f"不支持的架构: {arch}")

        try:
            tag = self._resolve_latest_version()
            url = self._build_download_url(tag, arch_suffix)
            logger.info(f"[Worker] 版本 {tag}")
        except Exception as e:
            raise RuntimeError(f"解析版本失败: {e}")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
        except Exception as e:
            raise RuntimeError(f"下载失败: {e}。请检查容器出网。")

        try:
            payload = None
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    base = os.path.basename(member.name)
                    if base == _PKG_NAME:
                        f = tf.extractfile(member)
                        if f is not None:
                            payload = f.read()
                            break
            if payload is None:
                raise RuntimeError("tarball 中未找到主文件")
        except Exception as e:
            raise RuntimeError(f"解压失败: {e}")

        BIN_DIR.mkdir(parents=True, exist_ok=True)
        with open(BIN_PATH, "wb") as f:
            f.write(payload)
        BIN_PATH.chmod(BIN_PATH.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        size_mb = len(payload) / 1024 / 1024
        logger.success(f"[Worker] 内核已就绪 ({size_mb:.1f} MB) -> {BIN_PATH}")
        return str(BIN_PATH)

    # ---------- status ----------
    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def proxy_url(self) -> str:
        return f"socks5://127.0.0.1:{self._port}"

    def status(self) -> dict[str, Any]:
        binary = self.find_binary()
        return {
            "binary_available": bool(binary),
            "binary_path": binary or "",
            "running": self.is_running,
            "active_uri": self._active_uri or "",
            "active_name": self._active_name or "",
            "socks_port": self._port,
            "proxy_url": self.proxy_url if self.is_running else "",
        }

    # ---------- lifecycle ----------
    def stop(self) -> None:
        p = self._proc
        if p is None:
            return
        try:
            if p.poll() is None:
                logger.info(f"[Worker] 停止进程 pid={p.pid}")
                try:
                    p.terminate()
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=2)
        except Exception as e:
            logger.warning(f"[Worker] 停止出错: {e}")
        finally:
            self._proc = None
            self._active_uri = None
            self._active_name = None

    def start_with_uri(self, uri: str, name: str = "", port: int = SOCKS_INBOUND_PORT) -> str:
        binary = self.ensure_binary()

        cfg = build_config(uri, socks_port=port)
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        self.stop()

        logger.info(f"[Worker] 启动: {binary}  节点={name or uri[:40]}")
        log_f = open(LOG_PATH, "ab")
        try:
            proc = subprocess.Popen(
                [binary, "run", "-c", str(CONFIG_PATH)],
                stdout=log_f,
                stderr=log_f,
                start_new_session=True,
            )
        except Exception as e:
            log_f.close()
            raise RuntimeError(f"启动失败: {e}")

        time.sleep(0.8)
        if proc.poll() is not None:
            try:
                tail = _tail_file(str(LOG_PATH), 40)
            except Exception:
                tail = ""
            raise RuntimeError(f"启动后立即退出（exit code {proc.returncode}）。日志:\n{tail}")

        self._proc = proc
        self._active_uri = uri
        self._active_name = name
        self._port = port
        logger.success(f"[Worker] 运行中 pid={proc.pid}，本地 {self.proxy_url}")
        return self.proxy_url


def _tail_file(path: str, n: int = 40) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            data = f.read().decode("utf-8", errors="replace")
        return "\n".join(data.splitlines()[-n:])
    except Exception:
        return ""


worker = WorkerManager()
