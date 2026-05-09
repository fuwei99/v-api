

import json
import time
from typing import Any, cast
import collections.abc

from src.core.errors import (
    VertexError,
    EmptyResponseError,
    InternalError,
    NotFoundError,
    InvalidArgumentError,
    RateLimitError, 
)
from src.utils.logger import get_logger
from src.utils.token_counter import calculate_usage_metadata
from .parser import (
    parse_upstream_data,
    parse_single_upstream_item,
    create_initial_state,
    finalize_state
)



logger = get_logger(__name__)


def _strip_sse_data_prefix(payload: str) -> str:
    """Remove SSE data prefixes before JSON parsing."""
    text = payload.strip()
    if not text:
        return text

    lines = text.splitlines()
    if lines and all(line.strip().startswith("data:") or not line.strip() for line in lines):
        return "\n".join(
            line.strip()[5:].lstrip()
            for line in lines
            if line.strip().startswith("data:")
        ).strip()

    if text.startswith("data:"):
        return text[5:].lstrip()

    return text


class StreamProcessor:
    """
    流式响应处理器。
    
    职责:
    1. 收集和聚合上游分块返回的流式数据。
    2. 使用 parser 解析并转换聚合后的数据结构。
    3. 将转换后的结果重新包装成 Gemini 标准的 SSE (Server-Sent Events) 格式返回给客户端。
    4. 在没有上游 Token 使用统计时，调用本地计数器估算 Token。
    """
    
    def __init__(self):
        logger.debug("初始化流处理器")
        
        
        self._actual_content_sent = False
        self._request_context: dict[str, Any] = {}
    
    def has_actual_content_sent(self) -> bool:
        """检查是否已发送实际文本内容"""
        return self._actual_content_sent
    
    def set_request_context(self, downstream_payload: dict[str, Any], upstream_payload: dict[str, Any]):
        """设置请求上下文"""
        logger.debug("设置流处理器请求上下文")
        self._request_context = {
            'downstream_payload': downstream_payload,
            'upstream_payload': upstream_payload
        }
    
    def _create_gemini_chunk(
        self,
        parts: list[dict[str, Any]],
        finish_reason: str | None,
        safety_ratings: list[dict[str, Any]],
        citation_metadata: dict[str, Any],
        grounding_metadata: dict[str, Any],
        candidate_index: int,
        prompt_feedback: dict[str, Any],
        usage_metadata: dict[str, Any],
        finish_message: str | None = None,
        token_count: int | None = None,
        avg_logprobs: float | None = None,
        logprobs_result: dict[str, Any] | None = None,
        create_time: str | None = None,
        model_version: str | None = None,
        response_id: str | None = None,
        model_status: dict[str, Any] | None = None,
    ) -> str:
        """根据聚合后的内容，创建一个包含完整上下文的Gemini格式SSE事件。"""
        candidate: dict[str, Any] = {
            "index": candidate_index
        }
        if finish_reason and isinstance(finish_reason, str):
            candidate["finishReason"] = finish_reason.upper()
        if finish_message:
            candidate["finishMessage"] = finish_message
        
        if parts:
            candidate["content"] = {
                "parts": parts,
                "role": "model"
            }
        
        
        if safety_ratings:
            candidate["safetyRatings"] = safety_ratings
        if citation_metadata:
            candidate["citationMetadata"] = citation_metadata
        if grounding_metadata:
            candidate["groundingMetadata"] = grounding_metadata
        if token_count is not None:
            candidate["tokenCount"] = token_count
        if avg_logprobs is not None:
            candidate["avgLogprobs"] = avg_logprobs
        if logprobs_result:
            candidate["logprobsResult"] = logprobs_result
            
        
        
        chunk: dict[str, Any] = {"candidates": [candidate]}
        
        
        if prompt_feedback:
            # 过滤误导性的 BLOCKED_REASON_UNSPECIFIED（Google API 默认占位值，不代表真的被拦截）
            pf = prompt_feedback.copy() if isinstance(prompt_feedback, dict) else prompt_feedback
            if isinstance(pf, dict) and pf.get("blockReason") == "BLOCKED_REASON_UNSPECIFIED":
                pf.pop("blockReason", None)
            if pf:
                chunk["promptFeedback"] = pf
        if usage_metadata:
            chunk["usageMetadata"] = usage_metadata
        if create_time:
            chunk["createTime"] = create_time
        if model_version:
            chunk["modelVersion"] = model_version
        if response_id:
            chunk["responseId"] = response_id
        if model_status:
            chunk["modelStatus"] = model_status
             
        return "data: " + json.dumps(chunk, ensure_ascii=False, separators=(',', ':')) + "\n\n"

    async def process_stream(
        self,
        response_iterator: collections.abc.AsyncIterator[str],
        model: str = "vertex-ai-proxy"
    ) -> collections.abc.AsyncGenerator[str, None]:
        """
        处理和包装来自上游的流式响应数据。
        支持实时流式解析，即解析一个 JSON 对象就返回一个 SSE 分块。
        """
        
        start_time = time.time()
        buffer = ""
        bracket_count = 0
        in_string = False
        escape_next = False
        
        state = create_initial_state()
        
        # 记录已发送给客户端的 Part 内容状态，用于计算增量
        # 格式: {part_index: { 'text': str, 'thought': bool }}
        sent_parts_content: dict[int, dict[str, Any]] = {}
        
        chunk_count = 0
        raw_chunks = []
        
        # 记录已处理到的 buffer 位置，避免重复扫描
        processed_pos = 0
        
        # 记录是否已发送过 finishReason
        finish_reason_sent = False
        
        logger.debug("开始流式处理上游数据")
        
        try:
            async for chunk in response_iterator:
                chunk_count += 1
                raw_chunks.append(chunk)
                buffer += chunk
                
                while processed_pos < len(buffer):
                    char = buffer[processed_pos]
                    
                    if escape_next:
                        escape_next = False
                    elif char == '\\':
                        escape_next = True
                    elif char == '"':
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            
                            if bracket_count == 0:
                                # 找到一个完整的对象
                                potential_obj_str = buffer[:processed_pos + 1]
                                # 尝试解析
                                try:
                                    # 预处理：去掉可能的前导 [ 或 ,
                                    clean_obj_str = _strip_sse_data_prefix(
                                        potential_obj_str.strip(' ,[]\n\r\t')
                                    )
                                    if clean_obj_str.startswith('{') and clean_obj_str.endswith('}'):
                                        item_dict = json.loads(clean_obj_str)
                                        parse_single_upstream_item(item_dict, state)
                                        
                                        current_result = finalize_state(state)
                                        current_parts = current_result.get("parts", [])
                                        
                                        # 计算增量部分
                                        delta_parts = []
                                        for i, p in enumerate(current_parts):
                                            p_text = p.get("text", "")
                                            is_thought = p.get("thought", False)
                                            
                                            if i not in sent_parts_content:
                                                delta_parts.append(p)
                                                sent_parts_content[i] = {"text": p_text, "thought": is_thought}
                                            else:
                                                old_content = sent_parts_content[i]["text"]
                                                if isinstance(p_text, str) and isinstance(old_content, str):
                                                    if p_text.startswith(old_content):
                                                        new_text = p_text[len(old_content):]
                                                        if new_text:
                                                            delta_part = p.copy()
                                                            delta_part["text"] = new_text
                                                            delta_parts.append(delta_part)
                                                    elif p_text:
                                                        delta_parts.append(p)

                                                    if p_text:
                                                        sent_parts_content[i]["text"] = (
                                                            p_text if p_text.startswith(old_content)
                                                            else old_content + p_text
                                                        )
                                                elif p != sent_parts_content[i].get("part"):
                                                    delta_parts.append(p)
                                                    sent_parts_content[i]["part"] = p
                                        
                                        # 如果有内容更新，或者有结束标志且还没发过，就发送
                                        has_finish = current_result.get("finish_reason") is not None
                                        
                                        if delta_parts or (has_finish and not finish_reason_sent):
                                            if has_finish:
                                                finish_reason_sent = True
                                                
                                            sse_chunk = self._create_gemini_chunk(
                                                parts=delta_parts,
                                                finish_reason=current_result.get("finish_reason"),
                                                safety_ratings=current_result.get("safety_ratings", []),
                                                citation_metadata=current_result.get("citation_metadata", {}),
                                                grounding_metadata=current_result.get("grounding_metadata", {}),
                                                candidate_index=current_result.get("candidate_index", 0),
                                                prompt_feedback=current_result.get("prompt_feedback", {}),
                                                usage_metadata=current_result.get("usage_metadata", {}),
                                                finish_message=current_result.get("finish_message"),
                                                token_count=current_result.get("token_count"),
                                                avg_logprobs=current_result.get("avg_logprobs"),
                                                logprobs_result=current_result.get("logprobs_result"),
                                                create_time=current_result.get("create_time"),
                                                model_version=current_result.get("model_version"),
                                                response_id=current_result.get("response_id"),
                                                model_status=current_result.get("model_status")
                                            )
                                            yield sse_chunk
                                            self._actual_content_sent = True
                                            
                                        # 成功处理后，截断 buffer
                                        buffer = buffer[processed_pos + 1:]
                                        processed_pos = -1 # 后面会 +1 变成 0
                                except json.JSONDecodeError:
                                    pass
                                except Exception as e:
                                    logger.warning(f"解析单个流对象失败: {e}")
                    
                    processed_pos += 1
            
            # 如果结束了还没发过任何内容（比如只有 usage 或已完成标志），补发一个结束包
            if not finish_reason_sent and not state["has_error"]:
                final_res = finalize_state(state)
                
                # 尝试补全 usage_metadata
                usage_metadata = final_res.get("usage_metadata", {})
                if not usage_metadata and self._request_context:
                    try:
                        downstream_payload = self._request_context.get('downstream_payload', {})
                        prompt_contents = []
                        if 'gemini_payload' in downstream_payload:
                            gemini_payload = downstream_payload['gemini_payload']
                            if isinstance(gemini_payload, dict) and 'contents' in gemini_payload:
                                prompt_contents = cast(list[dict[str, Any]], gemini_payload['contents'])
                        
                        usage_metadata = await calculate_usage_metadata(
                            prompt_contents=prompt_contents,
                            response_parts=final_res.get("parts", []),
                            request_context=self._request_context
                        )
                    except Exception:
                        pass

                yield self._create_gemini_chunk(
                    parts=[], # 既然之前可能已经发过内容，这里只发元数据
                    finish_reason=final_res.get("finish_reason", "STOP"),
                    usage_metadata=usage_metadata,
                    safety_ratings=final_res.get("safety_ratings", []),
                    citation_metadata=final_res.get("citation_metadata", {}),
                    grounding_metadata=final_res.get("grounding_metadata", {}),
                    candidate_index=final_res.get("candidate_index", 0),
                    prompt_feedback=final_res.get("prompt_feedback", {}),
                    finish_message=final_res.get("finish_message"),
                    token_count=final_res.get("token_count"),
                    avg_logprobs=final_res.get("avg_logprobs"),
                    logprobs_result=final_res.get("logprobs_result"),
                    create_time=final_res.get("create_time"),
                    model_version=final_res.get("model_version"),
                    response_id=final_res.get("response_id"),
                    model_status=final_res.get("model_status")
                )

            # 最后检查是否有未处理的错误
            if state["has_error"] and not self._actual_content_sent:
                error_obj = state.get("error_obj")
                if error_obj:
                    raise error_obj
                else:
                    raise InternalError(message=state["error_message"])

            # 流处理完成后保存所有原始数据块
            from src.utils.payload_logger import save_response
            save_response(raw_chunks)
            
            process_time = time.time() - start_time
            logger.success(f"流式响应处理完成: 耗时={process_time:.3f}s, 共处理 {chunk_count} 个原始数据块")
            
        except VertexError:
            raise
        except Exception as e:
            logger.error(f"流处理未知错误: {e}")
            raise InternalError(message=f"Unknown stream processing error: {e}")


def get_stream_processor() -> StreamProcessor:
    """创建流处理器实例"""
    return StreamProcessor()
