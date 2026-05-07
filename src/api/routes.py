

import json
import time
from typing import Any, cast
import collections.abc
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.core import MODELS_CONFIG_FILE
from src.core.errors import (
    VertexError,
    InvalidArgumentError,
    InternalError,
)
from src.api.vertex_client import VertexAIClient
from src.core.auth import api_key_manager
from src.utils.logger import get_logger, set_request_id


THINKING_CONFIG_MAP: dict[str, dict[str, Any]] = {
    "gemini-3.1-flash-image-preview": {
        "thinkingConfig": {
            "thinkingLevel": "HIGH",
            "includeThoughts": True,
        },
    },
    "gemini-3.1-flash-image-preview-nothinking": {
        "thinkingConfig": {
            "thinkingLevel": "MINIMAL",
            "includeThoughts": True,
        },
    },
    "gemini-2.5-pro-nothinking": {
        "temperature": 1,
        "topP": 0.95,
        "maxOutputTokens": 65535,
        "thinkingConfig": {
            "thinkingBudget": 128,
            "includeThoughts": True,
        },
    },
    "gemini-3-flash-preview-nothinking": {
        "temperature": 1,
        "topP": 0.95,
        "maxOutputTokens": 65535,
        "thinkingConfig": {
            "thinkingLevel": "MINIMAL",
            "includeThoughts": True,
        },
    },
    "gemini-3.1-pro-preview-low": {
        "temperature": 1,
        "topP": 0.95,
        "maxOutputTokens": 65535,
        "thinkingConfig": {
            "thinkingLevel": "LOW",
            "includeThoughts": True,
        },
    },
    "gemini-3.1-pro-preview-high": {
        "temperature": 1,
        "topP": 0.95,
        "maxOutputTokens": 65535,
        "thinkingConfig": {
            "thinkingLevel": "HIGH",
            "includeThoughts": True,
        },
    },
}


def _resolve_model_and_config(model: str, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    解析带后缀的模型名，去掉后缀得到真实模型名，并将对应的 thinkingConfig 注入到 payload 中。
    """
    if model not in THINKING_CONFIG_MAP:
        return model, body

    preset = THINKING_CONFIG_MAP[model]
    real_model = model.replace("-nothinking", "").replace("-low", "").replace("-high", "")

    body = body.copy()
    gen_config = body.get("generationConfig", {})
    if isinstance(gen_config, dict):
        gen_config = gen_config.copy()
    else:
        gen_config = {}

    for k, v in preset.items():
        if k == "thinkingConfig":
            existing_tc = gen_config.get("thinkingConfig", {})
            if isinstance(existing_tc, dict):
                merged_tc = {**existing_tc, **v}
                gen_config["thinkingConfig"] = merged_tc
            else:
                gen_config["thinkingConfig"] = v
        else:
            gen_config.setdefault(k, v)

    body["generationConfig"] = gen_config
    return real_model, body


logger = get_logger(__name__)


def extract_api_key_from_request(request: Request) -> str | None:
    """
    从请求中提取API密钥
    支持三种方式（按优先级）：
    1. Authorization: Bearer <key> (OpenAI 标准 Header)
    2. x-goog-api-key: <key> (Google/Gemini 标准 Header)
    3. ?key=<key> (Google/Gemini 标准 Query Param)
    """
    
    
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    
    goog_api_key = request.headers.get("x-goog-api-key")
    if goog_api_key:
        return goog_api_key.strip()
        
    
    query_key = request.query_params.get("key")
    if query_key:
        return query_key.strip()

    return None


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API密钥认证中间件"""

    def __init__(
        self,
        app: ASGIApp,
        excluded_paths: list[str] | None = None,
        excluded_prefixes: list[str] | None = None,
    ):
        super().__init__(app)
        self.excluded_paths: list[str] = excluded_paths or ["/", "/health"]
        self.excluded_prefixes: list[str] = excluded_prefixes or []

    async def dispatch(self, request: Request, call_next: collections.abc.Callable[[Request], collections.abc.Awaitable[Any]]):
        
        set_request_id()
        
        path = request.url.path
        method = request.method
        client_ip = request.client.host if request.client else "unknown"
        
        logger.debug(f"收到请求: {method} {path} from {client_ip}")

        
        if self.excluded_paths and path in self.excluded_paths:
            logger.debug(f"路径 {path} 在排除列表中，跳过认证")
            return await call_next(request)

        if any(path.startswith(prefix) for prefix in self.excluded_prefixes):
            logger.debug(f"路径 {path} 命中排除前缀，跳过认证")
            return await call_next(request)

        
        api_key = extract_api_key_from_request(request)
        if not api_key:
            logger.warning(f"请求 {path} 缺少 API 密钥")
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": 401,
                        "message": "Method doesn't allow unregistered callers (callers without established identity). Please use API Key or other form of API consumer identity to call this API.",
                        "status": "UNAUTHENTICATED"
                    }
                }
            )

        
        if not api_key_manager.validate_key(api_key):
            logger.warning(f"请求 {path} 使用了无效的 API 密钥: {api_key[:8]}...")
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": 400,
                        "message": "API key not valid. Please pass a valid API key.",
                        "status": "INVALID_ARGUMENT"
                    }
                }
            )

        
        request.state.api_key = api_key
        logger.debug(f"API 密钥验证成功: {api_key[:8]}...")

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(f"{method} {path} - {response.status_code} ({process_time:.3f}s)")
        
        return response


def create_app(vertex_client: VertexAIClient) -> FastAPI:
    """
    创建并配置 FastAPI 应用程序实例。
    
    参数:
        vertex_client: 用于处理 Vertex AI 请求的客户端实例
        
    功能:
        1. 实例化 FastAPI。
        2. 注册 APIKeyMiddleware 进行鉴权。
        3. 配置 CORS 跨域。
        4. 注册全局异常处理器（VertexError, Exception）。
        5. 定义基础端点（/, /health, /v1beta/models）。
        6. 定义 Gemini 兼容生成接口（streamGenerateContent, generateContent）。
    """
    logger.info("创建 FastAPI 应用")
    
    app = FastAPI(
        title="Vertex AI Proxy (Anonymous)",
        description="Vertex AI 代理服务，兼容 Gemini API",
        version="1.1.0"
    )

    
    logger.debug("添加中间件")
    app.add_middleware(
        APIKeyMiddleware,
        excluded_paths=["/", "/health", "/admin", "/favicon.ico"],
        excluded_prefixes=["/api/admin/", "/admin/", "/static/"],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    from src.api.admin import router as admin_router
    app.include_router(admin_router)

    
    
    @app.exception_handler(VertexError)
    async def vertex_exception_handler(request: Request, exc: VertexError):  
        
        logger.error(f"VertexError: {exc.message} (code={exc.code}, status={exc.status})")
        return JSONResponse(
            status_code=exc.code,
            content=exc.to_Dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):  
        
        logger.error(f"Unhandled Exception: {exc}", exc_info=True)
        error = InternalError(message=str(exc))
        return JSONResponse(
            status_code=500,
            content=error.to_Dict(),
        )

    

    async def root() -> RedirectResponse:
        """根路径跳转到管理面板。"""
        logger.debug("处理根路径请求")
        return RedirectResponse(url="/admin", status_code=302)
    app.get("/")(root)

    async def favicon() -> Response:
        return Response(status_code=204)
    app.get("/favicon.ico")(favicon)

    async def health_check() -> dict[str, str | int]:
        """健康检查端点"""
        logger.debug("处理健康检查请求")
        api_keys_count = len(api_key_manager.api_keys)
        logger.debug(f"当前加载的 API 密钥数量: {api_keys_count}")
        return {
            "status": "healthy",
            "timestamp": int(time.time()),
            "api_keys_loaded": api_keys_count
        }
    app.get("/health")(health_check)
    
    async def list_models() -> dict[str, Any]:
        """返回可用模型列表 (兼容 Gemini API 格式)"""
        logger.debug("处理模型列表请求")
        models: list[str] = _load_models_config()
        logger.debug(f"返回 {len(models)} 个可用模型")
        
        # 兼容 Gemini API 格式: 使用 "models" 键，且模型 ID 带有 "models/" 前缀
        return {
            "models": [
                {
                    "name": f"models/{m}",
                    "version": "unknown",
                    "displayName": m,
                    "description": f"Vertex AI Proxy Model: {m}",
                    "supportedGenerationMethods": ["generateContent", "streamGenerateContent"]
                }
                for m in models
            ]
        }
    app.get("/v1beta/models")(list_models)

    

    async def stream_generate_content(model: str, request: Request) -> StreamingResponse | JSONResponse:
        """Gemini 格式的流式生成接口"""
        logger.info(f"收到流式生成请求: 模型={model}")
        
        try:
            body_any = await request.json()
        except json.JSONDecodeError as e:
            raise InvalidArgumentError(f"Invalid JSON in request body: {e}")

        
        if not isinstance(body_any, dict):
                raise InvalidArgumentError("Request body must be a JSON object")
        body: dict[str, Any] = cast(dict[str, Any], body_any)
        
        logger.debug(f"请求体大小: {len(str(body))} 字符")
        
        
        logger.debug_json("下游请求体", body)

        real_model, body = _resolve_model_and_config(model, body)

        async def stream_generator():
            chunk_count = 0
            try:
                async for chunk in vertex_client.stream_chat(
                    model=real_model, gemini_payload=body
                ):
                    chunk_count += 1
                    yield chunk
            except VertexError as e:
                logger.error(f"流式生成 Vertex 错误: {e.message}")
                
                yield e.to_sse()
            except Exception as e:
                logger.error(f"流式生成未知错误: {e}")
                
                error = InternalError(message=str(e))
                yield error.to_sse()
            finally:
                logger.debug(f"流式生成完成，共发送 {chunk_count} 个数据块")

        return StreamingResponse(stream_generator(), media_type="application/json")
    app.post("/v1beta/models/{model}:streamGenerateContent", response_model=None)(stream_generate_content)

    async def generate_content(model: str, request: Request) -> JSONResponse | dict[str, Any]:
        """Gemini 格式的非流式生成接口"""
        logger.info(f"收到非流式生成请求: 模型={model}")
        
        try:
            body_any = await request.json()
        except json.JSONDecodeError as e:
            raise InvalidArgumentError(f"Invalid JSON in request body: {e}")

        if not isinstance(body_any, dict):
                raise InvalidArgumentError("Request body must be a JSON object")
        body: dict[str, Any] = cast(dict[str, Any], body_any)
        
        logger.debug(f"请求体大小: {len(str(body))} 字符")
        
        
        logger.debug_json("下游请求体", body)

        real_model, body = _resolve_model_and_config(model, body)

        start_time = time.time()
        
        
        response = await vertex_client.complete_chat(
            model=real_model,
            gemini_payload=body
        )
        
        process_time = time.time() - start_time
        logger.success(f"非流式生成完成: 模型={model}, 耗时={process_time:.3f}s")
        
        return response
    app.post("/v1beta/models/{model}:generateContent", response_model=None)(generate_content)
    
    logger.info("FastAPI 应用创建完成")
    return app




def _load_models_config() -> list[str]:
    """加载模型配置"""
    try:
        with open(MODELS_CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return cast(list[str], config.get('models', []))
    except Exception:
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-2.0-pro-exp-02-05", "gemini-2.5-flash"]
