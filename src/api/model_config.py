

import json
import time
from typing import Any, cast

from src.core import MODELS_CONFIG_FILE
from ..utils.logger import get_logger
from ..utils.string_utils import snake_to_camel


logger = get_logger(__name__)


class ModelConfigBuilder:
    """
    模型配置构建器。
    
    职责：
    1. 模型别名映射：将客户端传入的模型名（如 gemini-pro）映射为后端真实的接口名。
    2. 配置合并：合并全局配置、请求体中的配置以及 API 调用参数。
    3. 格式转换：将 Python 的 snake_case 参数名转换为 Google API 要求的 camelCase 格式。
    4. 预设安全设置：生成默认全开（BLOCK_NONE）的安全过滤配置。
    """
    
    _cached_map: dict[str, str] | None = None
    _last_load_time: float = 0
    
    def __init__(self) -> None:
        
        from src.utils.logger import get_request_id
        if not get_request_id():
            logger.info("模型配置构建器初始化完成", extra={
                "model_count": len(self._get_model_map())
            })
    
    def _get_model_map(self) -> dict[str, str]:
        
        current_time = time.time()
        if ModelConfigBuilder._cached_map is not None and current_time - ModelConfigBuilder._last_load_time < 60:
            return ModelConfigBuilder._cached_map
            
        try:
            with open(MODELS_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                ModelConfigBuilder._cached_map = cast(dict[str, str], config.get('alias_map', {}))
                ModelConfigBuilder._last_load_time = current_time
                logger.debug("模型配置文件加载成功", extra={
                    "config_file": MODELS_CONFIG_FILE,
                    "alias_count": len(ModelConfigBuilder._cached_map)
                })
        except Exception as e:
            logger.warning(f"模型配置文件加载失败，使用空配置", extra={
                "config_file": MODELS_CONFIG_FILE,
                "error": str(e)
            })
            if ModelConfigBuilder._cached_map is None:
                ModelConfigBuilder._cached_map = {}
        
        return ModelConfigBuilder._cached_map or {}
    
    def parse_model_name(self, model: str) -> str:
        """
        解析模型名称，返回 backend_model
        """
        return self._get_model_map().get(model, model)
    
    def build_generation_config(
        self,
        gen_config: dict[str, Any],
        gemini_payload: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """构建生成配置"""
        
        final_config = gen_config.copy()

        from src.core.config import load_config
        app_config = load_config()
        drop_max_tokens = app_config.get("drop_max_tokens", False)
        
        
        if gemini_payload:
            user_gen_config_raw = gemini_payload.get('generationConfig', {}) or gemini_payload.get('generation_config', {})
            if user_gen_config_raw:
                user_gen_config: dict[str, Any] = {}
                
                if hasattr(user_gen_config_raw, 'model_dump'):
                     user_gen_config = user_gen_config_raw.model_dump(exclude_none=True)
                elif isinstance(user_gen_config_raw, dict):
                     user_gen_config = cast(dict[str, Any], user_gen_config_raw)
                
                if user_gen_config:
                    final_config.update(user_gen_config)
                    if drop_max_tokens:
                        final_config.pop('max_output_tokens', None)
                        final_config.pop('maxOutputTokens', None)

        
        for k, v in kwargs.items():
            
            final_config[k] = v

        
        return self._convert_to_gemini_format(final_config)

    def _convert_to_gemini_format(self, config: dict[str, Any]) -> dict[str, Any]:
        """将 snake_case 配置转换为 camelCase"""
        converted: dict[str, Any] = {}
        for k, v in config.items():
            camel_key = snake_to_camel(k)
            
            
            if camel_key == "thinkingConfig" and isinstance(v, dict):
                thinking_config: dict[str, Any] = cast(dict[str, Any], v).copy()
                if "thinkingLevel" in thinking_config:
                    
                    level = thinking_config["thinkingLevel"]
                    if isinstance(level, str):
                        thinking_config["thinkingLevel"] = level.upper()
                converted[camel_key] = thinking_config
            elif camel_key == "topK" and isinstance(v, (int, float)):
                
                converted[camel_key] = min(63, int(v))
            else:
                converted[camel_key] = v
                
        return converted
    
    def build_safety_settings(self) -> list[dict[str, str]]:
        """构建安全设置"""
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
        ]
