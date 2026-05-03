

import json
import shutil
from datetime import datetime, timedelta
from typing import Any, cast
from pathlib import Path


class ErrorSnapshotManager:
    """
    错误快照管理器。
    
    当系统发生转换错误、上游 API 报错或空响应时，该组件负责将当时的现场环境
    （原始请求、转换后的请求、上游响应、时间戳及元数据）保存到磁盘。
    
    主要功能：
    1. 自动创建按日期组织的存储目录。
    2. 提供快照持久化、列表查询及内容检索。
    3. 自动清理机制：支持基于保留天数（max_age_days）和最大数量（max_snapshots）的旧快照清理。
    """
    
    def __init__(
        self,
        base_dir: str = "errors",
        max_snapshots: int = 100,
        max_age_days: int = 7,
        compress_old: bool = True
    ):
        """
        初始化错误快照管理器
        
        Args:
            base_dir: 快照存储基础目录
            max_snapshots: 最大保留快照数量
            max_age_days: 快照最大保留天数
            compress_old: 是否压缩旧快照
        """
        self.base_dir = Path(base_dir)
        self.max_snapshots = max_snapshots
        self.max_age_days = max_age_days
        self.compress_old = compress_old
        
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_snapshot(
        self,
        downstream_payload: dict[str, Any],
        upstream_payload: dict[str, Any],
        upstream_response: str,
        error_type: str = "request_error",
        metadata: dict[str, Any] | None = None
    ) -> str | None:
        """
        保存错误快照
        
        Args:
            downstream_payload: 接收到的原始请求
            upstream_payload: 发送给上游的请求
            upstream_response: 上游返回的响应
            error_type: 错误类型
            metadata: 额外元数据
        
        Returns:
            快照目录路径，失败返回 None
        """
        try:
            
            timestamp = datetime.now()
            date_dir = timestamp.strftime("%Y-%m-%d")
            time_str = timestamp.strftime("%H%M%S_%f")
            
            
            snapshot_name = f"{error_type}_{time_str}"
            snapshot_dir = self.base_dir / date_dir / snapshot_name
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            
            self._save_json(
                snapshot_dir / "1_downstream_request.json",
                downstream_payload
            )
            
            
            self._save_json(
                snapshot_dir / "2_upstream_request.json",
                upstream_payload
            )
            
            
            resp_ext = self._save_response(
                snapshot_dir / "3_upstream_response",
                upstream_response
            )
            
            
            summary: dict[str, Any] = {
                "timestamp": timestamp.isoformat(),
                "error_type": error_type,
                "snapshot_id": snapshot_name,
                "files": [
                    "1_downstream_request.json",
                    "2_upstream_request.json",
                    f"3_upstream_response{resp_ext}"
                ],
                "metadata": metadata or {}
            }
            
            self._save_json(snapshot_dir / "summary.json", summary)
            
            
            self._cleanup_old_snapshots()
            
            return str(snapshot_dir)
            
        except Exception:
            return None
    
    def _save_json(self, path: Path, data: Any):
        """保存 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                f.write(str(data))
    
    def _save_response(self, path_base: Path, response: str) -> str:
        """保存响应内容，自动检测格式，返回文件后缀"""
        if self._is_json(response):
            try:
                data = json.loads(response)
                self._save_json(path_base.with_suffix('.json'), data)
                return ".json"
            except json.JSONDecodeError:
                pass
        
        
        with open(path_base.with_suffix('.txt'), 'w', encoding='utf-8') as f:
            f.write(str(response))
        return ".txt"
    
    def _is_json(self, text: str) -> bool:
        """检查是否为 JSON 格式"""
        
        
        
        
        
        
        if not text:
            return False
            
        
        text_str = str(text).strip()
        return text_str.startswith('{') or text_str.startswith('[')
    
    def _cleanup_old_snapshots(self):
        """清理旧快照"""
        try:
            
            date_dirs = sorted(
                [d for d in self.base_dir.iterdir() if d.is_dir()],
                key=lambda x: x.name,
                reverse=True
            )
            
            
            cutoff_date = datetime.now() - timedelta(days=self.max_age_days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            
            for date_dir in date_dirs:
                if date_dir.name < cutoff_str:
                    
                    shutil.rmtree(date_dir)
            
            
            all_snapshots: list[dict[str, Any]] = []
            for date_dir in self.base_dir.iterdir():
                if date_dir.is_dir():
                    for snapshot_dir in date_dir.iterdir():
                        if snapshot_dir.is_dir():
                            summary_file = snapshot_dir / "summary.json"
                            if summary_file.exists():
                                try:
                                    with open(summary_file, 'r') as f:
                                        summary = json.load(f)
                                    all_snapshots.append({
                                        'path': snapshot_dir,
                                        'timestamp': str(summary.get('timestamp', ''))
                                    })
                                except Exception:
                                    pass
            
            
            
            def sort_key(x: dict[str, Any]) -> str:
                return str(x['timestamp'])

            all_snapshots.sort(key=sort_key, reverse=True)
            
            for snapshot in all_snapshots[self.max_snapshots:]:
                snapshot_path = cast(Path, snapshot['path'])
                shutil.rmtree(snapshot_path)
            
            
            for date_dir in self.base_dir.iterdir():
                if date_dir.is_dir() and not any(date_dir.iterdir()):
                    date_dir.rmdir()
                    
        except Exception:
            pass
    
    def list_snapshots(
        self,
        error_type: str | None = None,
        limit: int = 20
    ) -> list[dict[str, Any]]:
        """
        列出快照
        
        Args:
            error_type: 过滤错误类型
            limit: 返回数量限制
        
        Returns:
            快照摘要列表
        """
        snapshots: list[dict[str, Any]] = []
        
        try:
            for date_dir in sorted(self.base_dir.iterdir(), reverse=True):
                if not date_dir.is_dir():
                    continue
                    
                for snapshot_dir in sorted(date_dir.iterdir(), reverse=True):
                    if not snapshot_dir.is_dir():
                        continue
                    
                    summary_file = snapshot_dir / "summary.json"
                    if not summary_file.exists():
                        continue
                    
                    try:
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                        
                        if error_type and summary.get('error_type') != error_type:
                            continue
                        
                        summary['path'] = str(snapshot_dir)
                        snapshots.append(summary)
                        
                        if len(snapshots) >= limit:
                            return snapshots
                            
                    except Exception:
                        continue
                        
        except Exception:
            pass
        
        return snapshots
    
    def get_snapshot(self, snapshot_path: str) -> dict[str, Any] | None:
        """
        获取完整快照内容
        
        Args:
            snapshot_path: 快照目录路径
        
        Returns:
            快照完整内容
        """
        try:
            snapshot_dir = Path(snapshot_path)
            if not snapshot_dir.exists():
                return None
            
            result: dict[str, Any] = {}
            
            for file in snapshot_dir.iterdir():
                if file.suffix == '.json':
                    with open(file, 'r', encoding='utf-8') as f:
                        result[file.stem] = json.load(f)
                elif file.suffix == '.txt':
                    with open(file, 'r', encoding='utf-8') as f:
                        result[file.stem] = f.read()
            
            return result
            
        except Exception:
            return None


_snapshot_manager: ErrorSnapshotManager | None = None


def _get_manager() -> ErrorSnapshotManager:
    """获取或创建快照管理器"""
    global _snapshot_manager
    if _snapshot_manager is None:
        try:
            from src.core.config import load_config
            config = load_config()
            base_dir = config.get("error_dir", "errors")
        except Exception:
            base_dir = "errors"
        
        _snapshot_manager = ErrorSnapshotManager(base_dir=base_dir)
    
    return _snapshot_manager



def save_error_snapshot(
    downstream_payload: dict[str, Any],
    upstream_payload: dict[str, Any],
    upstream_response: str,
    error_type: str = "request_error"
) -> str | None:
    """
    保存请求错误快照
    
    这是主要的对外接口，保持与原有 API 兼容。
    
    Args:
        downstream_payload: 接收到的原始请求 (Gemini 格式)
        upstream_payload: 最终发送给 Google 的 Payload
        upstream_response: 上游返回的原始错误信息
        error_type: 错误类别 (如 http_400, transform_error 等)
    
    Returns:
        快照目录路径，失败返回 None
    """
    manager = _get_manager()
    return manager.save_snapshot(
        downstream_payload=downstream_payload,
        upstream_payload=upstream_payload,
        upstream_response=upstream_response,
        error_type=error_type
    )


def list_error_snapshots(
    error_type: str | None = None,
    limit: int = 20
) -> list[dict[str, Any]]:
    """
    列出错误快照
    
    Args:
        error_type: 过滤错误类型
        limit: 返回数量限制
    
    Returns:
        快照摘要列表
    """
    manager = _get_manager()
    return manager.list_snapshots(error_type=error_type, limit=limit)


def get_error_snapshot(snapshot_path: str) -> dict[str, Any] | None:
    """
    获取完整错误快照
    
    Args:
        snapshot_path: 快照目录路径
    
    Returns:
        快照完整内容
    """
    manager = _get_manager()
    return manager.get_snapshot(snapshot_path)
