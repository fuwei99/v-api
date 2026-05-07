
import json
import os
from typing import Any, cast
from pathlib import Path
from src.utils.logger import get_logger
from .types import AppConfig


logger = get_logger(__name__)

CONFIG_FILE = str(Path(__file__).parent.parent.parent / "config" / "config.json")

def load_config() -> dict[str, Any]:
    """
    加载配置。优先级：
    1. 环境变量 CONFIG（JSON 字符串）—— 适合 Docker 部署
    2. config/config.json 文件
    3. 代码内置默认值
    """
    default_config = AppConfig()

    # ── 优先：环境变量 CONFIG ──────────────────────────────────────
    # CONFIG 既支持 JSON 字符串，也支持指向 JSON 文件的路径。
    env_config_str = os.environ.get("CONFIG", "").strip()
    if env_config_str:
        try:
            env_config_path = Path(env_config_str)
            if env_config_path.exists() and env_config_path.is_file():
                with open(env_config_path, 'r', encoding='utf-8') as f:
                    env_config = json.load(f)
                config_source = str(env_config_path)
            else:
                env_config = json.loads(env_config_str)
                config_source = "CONFIG"

            config_dict = default_config.model_dump()
            config_dict.update(env_config)
            final_config = AppConfig(**config_dict)
            final_dict = final_config.model_dump()
            from src.utils.logger import get_request_id
            if not get_request_id():
                logger.info("配置已从环境变量 CONFIG 加载", extra={
                    "config_source": config_source,
                    "port_api": final_dict.get("port_api"),
                    "debug_mode": final_dict.get("debug")
                })
            return final_dict
        except Exception as e:
            logger.error(f"环境变量 CONFIG 解析失败，回退到配置文件: {e}")

    # ── 回退：config/config.json ───────────────────────────────────
    if not os.path.exists(CONFIG_FILE):
        logger.info("配置文件不存在，使用默认配置", extra={
            "config_file": CONFIG_FILE
        })
        return default_config.model_dump()
        
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
            
            config_dict = default_config.model_dump()
            config_dict.update(file_config)
            
            final_config = AppConfig(**config_dict)
            final_dict = final_config.model_dump()
            
            from src.utils.logger import get_request_id
            if not get_request_id():
                logger.info("配置文件加载成功", extra={
                    "config_file": CONFIG_FILE,
                    "port_api": final_dict.get("port_api"),
                    "debug_mode": final_dict.get("debug")
                })
            return final_dict
            
    except Exception as e:
        logger.error(f"配置文件加载失败，使用默认配置", extra={
            "config_file": CONFIG_FILE,
            "error": str(e)
        })
        return default_config.model_dump()

