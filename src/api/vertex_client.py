import asyncio
import json
from typing import Any, cast, AsyncGenerator
from src.core.config import load_config

from src.core.errors import (
    VertexError,
    AuthenticationError,
    RateLimitError,
    InternalError,
    parse_error_response,
    raise_for_status,
)
from src.utils.logger import get_logger

from .model_config import ModelConfigBuilder
from .transform import RequestTransformer, ResponseAggregator
from .network import NetworkClient

logger = get_logger(__name__)

class VertexAIClient:
    """
    Vertex AI 代理客户端，负责与上游 Google Vertex AI 接口进行通信。
    
    主要逻辑：
    1. 管理与 Google 的网络会话和 Recaptcha Token 获取。
    2. 实现请求体转换（Gemini -> Vertex 匿名格式）。
    3. 执行“探路请求”机制，激活 Token 会话以绕过认证拦截。
    4. 处理流式响应聚合与错误重试逻辑。
    """
    
    def __init__(self):
        logger.info("初始化 Vertex AI 客户端")
        
        self.config = load_config()
        self.max_retries = self.config.get("max_retries", 10)
        
        self.model_builder = ModelConfigBuilder()
        self.transformer = RequestTransformer(self.model_builder)
        self.aggregator = ResponseAggregator()
        self.network = NetworkClient()
        
        self.vertex_ai_anonymous_base_api = "https://cloudconsole-pa.clients6.google.com"
        
        logger.success("Vertex AI 客户端初始化完成")

    async def close(self):
        await self.network.close()

    async def complete_chat(self, model: str, gemini_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        _raw_image_response = kwargs.pop('_raw_image_response', False)
        return await self.aggregator.aggregate_stream(
            self.stream_chat(model, gemini_payload=gemini_payload, **kwargs),
            _raw_image_response=_raw_image_response
        )

    async def stream_chat(self, model: str, gemini_payload: dict[str, Any], **kwargs: Any) -> AsyncGenerator[str, Any]:
        logger.info(f"开始流式聊天请求: 模型={model}")
        config = load_config()
        pool: list[dict[str, Any]] = config.get("node_pool", [])
        if len(pool) < 2:
            async for chunk in self._stream_chat_inner(model, gemini_payload=gemini_payload, **kwargs):
                yield chunk
            return

        index = int(config.get("node_pool_index", 0) or 0) % len(pool)
        tried = 0
        last_error_chunk: str | None = None

        while tried < len(pool):
            node = pool[index]
            node_name = str(node.get("name") or node.get("raw_uri", "")[:40])
            logger.info(f"节点池：使用节点 [{index + 1}/{len(pool)}] {node_name}")
            await self._apply_pool_node(node, index)

            got_content = False
            failed = False
            async for chunk in self._stream_chat_inner(model, gemini_payload=gemini_payload, **kwargs):
                if '"error"' in chunk and not got_content:
                    last_error_chunk = chunk
                    failed = True
                    break
                got_content = True
                last_error_chunk = None
                yield chunk

            if not failed:
                return

            tried += 1
            index = (index + 1) % len(pool)
            logger.warning(f"节点失败，自动切换到节点 [{index + 1}/{len(pool)}]")

        if last_error_chunk:
            yield last_error_chunk
        else:
            yield InternalError(message="节点池所有节点均失败").to_sse()

    async def _apply_pool_node(self, node: dict[str, Any], index: int) -> None:
        """切换代理到节点池中指定节点，写入 config，network 会在新 Session 中动态读取。"""
        raw_uri = str(node.get("raw_uri") or "")
        name = str(node.get("name") or "")
        try:
            from src.api.admin import _activate_node_by_uri
            await _activate_node_by_uri(raw_uri, name, index)
            logger.info(f"节点池已激活节点 [{index + 1}]: {name or raw_uri[:40]}")
        except Exception as e:
            logger.warning(f"切换节点代理失败: {e}")

    async def _execute_single_attempt(
        self,
        session: Any,
        model: str,
        gemini_payload: dict[str, Any],
        recaptcha_token: str,
        attempt: int,
        kwargs: dict[str, Any],
        is_first_auth_attempt: bool = False
    ):
        dummy_original_body = {"variables": {}}
        new_variables = self.transformer.build_vertex_payload(
            model=model,
            gemini_payload=gemini_payload,
            original_body=dummy_original_body,
            kwargs=kwargs
        )['variables']
        
        new_variables["region"] = "global"
        new_variables["recaptchaToken"] = recaptcha_token

        new_body = {
            "querySignature": "2/l8eCsMMY49imcDQ/lwwXyL8cYtTjxZBF2dNqy69LodY=",
            "operationName": "StreamGenerateContentAnonymous",
            "variables": new_variables,
        }
        
        downstream_payload: dict[str, Any] = {
            "model": model,
            "gemini_payload": gemini_payload,
            "kwargs": {k: v for k, v in kwargs.items() if k != 'tools'},
            "attempt": attempt,
            "instance_id": "anonymous"
        }
        
        from src.stream import get_stream_processor
        stream_processor = get_stream_processor()
        stream_processor.set_request_context(downstream_payload, new_body)
        
        headers = {
            "referer": "https://console.cloud.google.com/",
            "Content-Type": "application/json",
        }
        
        url = f"{self.vertex_ai_anonymous_base_api}/v3/entityServices/AiplatformEntityService/schemas/AIPLATFORM_GRAPHQL:batchGraphql?key=AIzaSyCI-zsRP85UVOi0DjtiCwWBwQ1djDy741g&prettyPrint=false"
        
        logger.debug(f"准备发送请求到: {url[:50]}...")
        
        if attempt > 0 or not is_first_auth_attempt:
             logger.debug_json("发送 Vertex AI 请求体", new_body)
        
        from src.utils.payload_logger import save_fetch
        save_fetch(new_body)
        
        async for response in self.network.stream_request(session, 'POST', url, headers=headers, json_data=new_body):
            if response.status_code != 200:
                error_text = await response.aread()
                error_text_str = error_text.decode() if error_text else ""
                
                if is_first_auth_attempt and (response.status_code in[401, 403] or "Failed to verify action" in error_text_str):
                    logger.debug(f"上游服务返回预期内的首次认证失败: HTTP {response.status_code}")
                else:
                    logger.error(f"上游服务返回错误: HTTP {response.status_code}")
                    logger.debug_large("完整上游错误响应", error_text_str)
                
                if response.status_code in [401, 403] or "Failed to verify action" in error_text_str or "The caller does not have permission" in error_text_str:
                    raise AuthenticationError(
                        message=f"Authentication/Recaptcha failed: {error_text_str}",
                        details={"upstream_response": error_text_str},
                        upstream_response=error_text_str
                    )
                
                parsed_error = parse_error_response(error_text_str)
                if parsed_error:
                    parsed_error.upstream_response = error_text_str
                    raise parsed_error
                else:
                    raise raise_for_status(
                        code=response.status_code,
                        message=f"Upstream Error: {error_text_str}",
                        upstream_response=error_text_str
                    )
            
            logger.debug("开始处理流式响应")
            chunk_count = 0
            full_response_content: list[dict[str, Any]] =[]
            has_auth_error_in_stream = False
            
            async def line_iterator():
                async for line in response.aiter_lines():
                    decoded_line = line.decode('utf-8') if isinstance(line, bytes) else line
                    yield decoded_line

            try:
                async for sse_event in stream_processor.process_stream(line_iterator(), model=model):

                    chunk_count += 1
                    try:
                        chunk_str = str(sse_event)
                        if chunk_str.strip().startswith("data: "):
                             data_str = chunk_str.strip()[6:]
                             data_obj = json.loads(data_str)
                             full_response_content.append(data_obj)
                    except Exception:
                        pass
                    yield sse_event
            except VertexError as e:
                if isinstance(e, AuthenticationError) or "Failed to verify action" in str(e) or "The caller does not have permission" in str(e):
                    raise AuthenticationError(
                        message=f"Authentication/Recaptcha failed in parser: {e}",
                        details={"upstream_response": str(e)},
                        upstream_response=str(e)
                    )
                else:
                    raise e
            
            if full_response_content:
                 logger.debug_json("完整上游响应摘要", full_response_content)
            
            logger.success(f"流式响应处理完成，共处理 {chunk_count} 个数据块")
            return

    async def _stream_chat_inner(self, model: str, gemini_payload: dict[str, Any], **kwargs: Any) -> AsyncGenerator[str, Any]:
        """
        内部核心请求循环，实现了复杂的重试机制和 Token 会话激活。
        
        为什么需要这个函数？
        Google Vertex AI 的匿名接口具有严格的 Recaptcha 校验和会话激活机制。
        
        逻辑步骤：
        1. 获取 Recaptcha Token（如果过期或不存在）。
        2. 探路请求 (First Auth Attempt)：发送一个微小的“你好”请求。
           - 这个请求预期会触发 401 拦截或直接成功。
           - 关键在于此请求完成后，Google 内部会针对当前的 Session 和 Token 激活会话。
        3. 正式请求 (Official Attempt)：发送用户真实的业务请求。
        4. 重试逻辑 (Retry Logic)：
           - 如果遇到认证错误 (AuthenticationError)，重置 Token 并重试。
           - 如果遇到限流 (RateLimitError)，根据 retry-after 指示等待后重试。
           - 如果已开始发送流数据 (content_yielded)，则不再尝试重试，避免数据断层。
        """
        max_retries = self.max_retries
        content_yielded = False
        
        logger.debug(f"开始内部流式聊天，最大重试次数: {max_retries}")

        recaptcha_token = None
        is_first_auth_attempt = True
        attempt = 0
        
        session = self.network.create_session()
        try:
            while attempt <= max_retries:
                if not recaptcha_token:
                    pool = load_config().get("node_pool", [])
                    token_attempts = 1 if len(pool) > 1 else 3
                    recaptcha_token = await self.network.fetch_recaptcha_token(
                        session,
                        max_attempts=token_attempts
                    )
                    is_first_auth_attempt = True
                
                if not recaptcha_token:
                    if len(pool) > 1:
                        yield AuthenticationError(
                            "Could not fetch recaptcha token via current proxy."
                        ).to_sse()
                        return
                    if attempt == max_retries:
                        yield AuthenticationError("Could not fetch recaptcha token.").to_sse()
                        return
                    attempt += 1
                    await session.close()
                    session = self.network.create_session()
                    await asyncio.sleep(1)
                    continue
                
                if is_first_auth_attempt:
                    logger.info("发送轻量探路请求，激活 Token ...")
                    dummy_payload = {
                        "contents": [{"role": "user", "parts":[{"text": "你好"}]}]
                    }
                    dummy_model = "gemini-3.1-flash-lite-preview" 
                    
                    try:
                        async for _ in self._execute_single_attempt(
                            session, dummy_model, dummy_payload, recaptcha_token, attempt, {},
                            is_first_auth_attempt=True
                        ):
                            pass 
                        logger.debug("探路请求通过 (未触发 401拦截)")
                    except AuthenticationError:
                        logger.debug("探路请求被拦截 (符合预期，Token 会话已激活)")
                    except Exception as e:
                        logger.debug(f"探路请求发生其他异常: {e}")
                    
                    is_first_auth_attempt = False
                    continue

                logger.debug(f"尝试第 {attempt + 1}/{max_retries + 1} 次正式请求")
                
                try:
                    async for chunk in self._execute_single_attempt(
                        session, model, gemini_payload, recaptcha_token, attempt, kwargs,
                        is_first_auth_attempt=False
                    ):
                        yield chunk
                        content_yielded = True
                    
                    break
                
                except AuthenticationError as e:
                    logger.warning(f"正式请求发生认证/Recaptcha错误: {e.message}")
                    recaptcha_token = None
                    
                    if content_yielded:
                        logger.error("已产生内容，无法安全重试")
                        yield e.to_sse()
                        return
                    
                    if attempt < max_retries:
                        attempt += 1
                        await asyncio.sleep(1)
                        continue
                    else:
                        logger.error("重试次数耗尽")
                        yield e.to_sse()
                        return
                
                except RateLimitError as e:
                    logger.warning(f"限流错误: {e.message}")
                    if content_yielded or attempt >= max_retries:
                        yield e.to_sse()
                        return
                    
                    wait_time = e.retry_after if e.retry_after else min(30, 0)
                    logger.info(f"触发限流，等待 {wait_time}s 后重试 (第 {attempt + 1} 次重试)")
                    attempt += 1
                    await asyncio.sleep(wait_time)
                    continue
                
                except VertexError as e:
                    logger.error(f"Vertex 错误: {e.message}")
                    if not e.is_retryable or content_yielded or attempt >= max_retries:
                         yield e.to_sse()
                         return
                
                    wait_time = min(15, 0)
                    logger.info(f"触发可重试 Vertex 错误，等待 {wait_time:.1f}s 后重试 (第 {attempt + 1} 次重试)")
                    attempt += 1
                    await asyncio.sleep(wait_time)
                    continue
                
                except Exception as e:
                    logger.error(f"未预期的异常: {e}")
                    if content_yielded or attempt >= max_retries:
                        yield InternalError(message=f"Internal error: {e}").to_sse()
                        return
                    
                    attempt += 1
                    await asyncio.sleep(1)
                    continue
        finally:
            await session.close()
