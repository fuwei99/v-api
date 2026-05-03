import os
import threading
from typing import Any
from pathlib import Path
from .config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class APIKeyManager:
    """
    API 密钥管理器，负责从本地文件加载和验证客户端请求所携带的 sk- 密钥。
    
    文件格式要求：
    name:sk-xxxxxxxxxxxxxx
    
    功能：
    1. 线程安全地加载密钥文件。
    2. 校验密钥格式是否以 'sk-' 开头。
    3. 提供快速验证接口 validate_key。
    """

    def __init__(self, keys_file: str | None = None):
        logger.info("初始化 API 密钥管理器")
        
        self.keys_file: str = keys_file or str(Path(__file__).parent.parent.parent / "config" / "api_keys.txt")
        self.api_keys: set[str] = set()
        self.key_names: dict[str, str] = {}
        self._lock: threading.Lock = threading.Lock()
        
        logger.debug(f"API 密钥文件路径: {self.keys_file}")

    def load_keys(self) -> bool:
        logger.info("开始加载 API 密钥")
        
        try:
            with self._lock:
                self.api_keys.clear()
                self.key_names.clear()
            
            # 1. 从 config.json 加载单个密钥
            config = load_config()
            config_key = config.get("api_key")
            if config_key and isinstance(config_key, str) and config_key.startswith('sk-'):
                with self._lock:
                    self.api_keys.add(config_key)
                    self.key_names[config_key] = "config_file"
                logger.debug(f"从 config.json 加载密钥: config_file ({config_key[:8]}...)")

            # 2. 从 api_keys.txt 加载多个密钥
            if not os.path.exists(self.keys_file):
                logger.warning(f"API 密钥文件不存在: {self.keys_file}")
                # 如果 config.json 里有密钥，即便 txt 不存在也返回成功
                return len(self.api_keys) > 0

            with self._lock:
                valid_count = 0
                error_count = 0

                logger.debug(f"读取密钥文件: {self.keys_file}")
                with open(self.keys_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()

                        if not line or line.startswith('#'):
                            continue

                        parts = line.split(':', 2)
                        if len(parts) < 2:
                            logger.warning(f"第 {line_num} 行格式错误，跳过")
                            error_count += 1
                            continue

                        key_name = parts[0].strip()
                        api_key = parts[1].strip()

                        if not api_key.startswith('sk-'):
                            logger.warning(f"第 {line_num} 行密钥格式无效 ({key_name})，跳过")
                            error_count += 1
                            continue

                        self.api_keys.add(api_key)
                        self.key_names[api_key] = key_name
                        valid_count += 1
                        logger.debug(f"加载密钥: {key_name} ({api_key[:8]}...)")

            if valid_count > 0:
                logger.success(f"成功加载 {valid_count} 个 API 密钥")
            if error_count > 0:
                logger.warning(f"跳过 {error_count} 个无效条目")
                
            return True

        except Exception as e:
            logger.error(f"加载 API 密钥失败: {e}")
            return False

    def validate_key(self, api_key: str) -> bool:
        if not api_key:
            logger.debug("API 密钥为空")
            return False

        is_valid = api_key.strip() in self.api_keys
        if is_valid:
            key_name = self.key_names.get(api_key.strip(), 'unknown')
            logger.debug(f"API 密钥验证成功: {key_name} ({api_key[:8]}...)")
        else:
            logger.debug(f"API 密钥验证失败: {api_key[:8]}...")
            
        return is_valid

api_key_manager = APIKeyManager()
