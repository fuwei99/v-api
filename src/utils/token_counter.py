

import json
from typing import Any, cast
from src.utils.logger import get_logger
from src.core.config import load_config
from src.api.network import NetworkClient
from src.api.model_config import ModelConfigBuilder

logger = get_logger(__name__)

class TokenCounter:
    """
    Token 计数器。
    
    利用 Google Vertex AI 的 CountTokens API 接口，实现精准的 Token 数量统计。
    由于代理的上游接口有时不返回使用统计（usageMetadata），该工具类用于在响应结束后
    手动发起一次 CountTokens 请求，以补全消耗统计信息。
    """
    
    def __init__(self, network: NetworkClient | None = None) -> None:
        self.config = load_config()
        self.vertex_ai_anonymous_base_api = "https://cloudconsole-pa.clients6.google.com"
        self._api_key = "AIzaSyCI-zsRP85UVOi0DjtiCwWBwQ1djDy741g"
        self.network = network or NetworkClient()
        self.model_builder = ModelConfigBuilder()
        
    async def calculate_usage_metadata_async(
        self, 
        prompt_contents: list[dict[str, Any]], 
        response_parts: list[dict[str, Any]],
        model: str = "gemini-2.5-flash"
    ) -> dict[str, Any]:
        """
        异步计算完整的 usage metadata
        """
        try:
            def clean_contents(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
                cleaned = []
                for content in contents:
                    new_content = content.copy()
                    if "parts" in new_content:
                        new_parts = []
                        for part in new_content["parts"]:
                            new_part = {}
                            if "text" in part:
                                new_part["text"] = part["text"]
                            if "inlineData" in part:
                                new_part["inlineData"] = part["inlineData"]
                            if "fileData" in part:
                                new_part["fileData"] = part["fileData"]

                            
                            if "functionCall" in part:
                                func_call = part["functionCall"]
                                text_rep = f"Function Call: {func_call.get('name', 'unknown')}"
                                if "args" in func_call:
                                    try: text_rep += f" Args: {json.dumps(func_call['args'])}"
                                    except: text_rep += f" Args: {str(func_call['args'])}"
                                new_part["text"] = new_part.get("text", "") + "\n" + text_rep
                                
                            if "functionResponse" in part:
                                func_resp = part["functionResponse"]
                                text_rep = f"Function Response: {func_resp.get('name', 'unknown')}"
                                if "response" in func_resp:
                                    try: text_rep += f" Result: {json.dumps(func_resp['response'])}"
                                    except: text_rep += f" Result: {str(func_resp['response'])}"
                                new_part["text"] = new_part.get("text", "") + "\n" + text_rep

                            if new_part:
                                new_parts.append(new_part)
                        
                        if new_parts:
                            new_content["parts"] = new_parts
                        else:
                            new_content["parts"] = [{"text": " "}]
                    cleaned.append(new_content)

                
                merged = []
                for c in cleaned:
                    if not merged:
                        merged.append(c)
                    elif merged[-1].get("role") == c.get("role"):
                        merged[-1]["parts"].extend(c.get("parts", []))
                    else:
                        merged.append(c)
                
                if merged and merged[0].get("role") == "model":
                    merged.insert(0, {"role": "user", "parts": [{"text": " "}]})
                    
                return merged

            safe_prompt_contents = clean_contents(prompt_contents)
            safe_response_parts = response_parts
            
            return await self._calculate_usage_with_session(safe_prompt_contents, safe_response_parts, model, clean_contents)
                    
        except Exception as e:
            logger.error(f"计算 usage metadata 失败: {e}")
            return {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}

    async def _calculate_usage_with_session(
        self,
        safe_prompt_contents: list[dict[str, Any]],
        response_parts: list[dict[str, Any]],
        model: str,
        clean_contents_fn: Any
    ) -> dict[str, Any]:
        prompt_token_count = await self._count_tokens_with_session(safe_prompt_contents, model)
        
        full_contents = list(safe_prompt_contents)
        if response_parts:
            
            import copy
            full_contents = copy.deepcopy(safe_prompt_contents)
            
            
            
            model_reply = clean_contents_fn([{"parts": response_parts, "role": "model"}])
            
            if model_reply:
                
                
                pass
            
            
            if full_contents and full_contents[-1].get("role") == "model":
                full_contents[-1]["parts"].extend(response_parts)
            else:
                full_contents.append({"role": "model", "parts": response_parts})
            
            
            full_contents = clean_contents_fn(full_contents)
        
        total_token_count = await self._count_tokens_with_session(full_contents, model)
        
        if total_token_count < prompt_token_count:
            total_token_count = prompt_token_count
            
        candidates_token_count = total_token_count - prompt_token_count
        
        usage_metadata: dict[str, Any] = {
            "promptTokenCount": prompt_token_count,
            "candidatesTokenCount": candidates_token_count,
            "totalTokenCount": total_token_count
        }
        
        logger.debug(f"Token 计算结果: {usage_metadata}")
        return usage_metadata

    async def _count_tokens_with_session(self, contents: list[dict[str, Any]], model: str) -> int:
        try:
            target_model = self.model_builder.parse_model_name(model)

            url = f"{self.vertex_ai_anonymous_base_api}/v3/entityServices/AiplatformEntityService/schemas/AIPLATFORM_GRAPHQL:batchGraphql?key={self._api_key}&prettyPrint=false"
            
            async with self.network.create_session() as session:
                recaptcha_token = await self.network.fetch_recaptcha_token(session)
                if not recaptcha_token:
                    return 0
                
                
                if target_model.startswith("models/"):
                    target_model = target_model[7:]

                payload = {
                    "requestContext": {
                        "clientVersion": "boq_cloud-boq-clientweb-vertexaistudio_20260402.09_p0",
                        "pagePath": "/vertex-ai/studio/multimodal",
                        "jurisdiction": "global",
                        "localizationData": {
                            "locale": "zh_CN",
                            "timezone": "Asia/Shanghai"
                        }
                    },
                    "querySignature": "2/mENOSldfC+HZM+tGhVuJLrl8M6gEyK3HRjUKuA5AM58=",
                    "operationName": "CountTokens",
                    "variables": {
                        "contents": contents,
                        "endpoint": "",
                        "model": target_model,
                        "region": "global",
                        "recaptchaToken": recaptcha_token
                    }
                }
                
                headers = {
                    "referer": "https://console.cloud.google.com/",
                    "Content-Type": "application/json",
                }
                
                logger.debug_json("CountTokens 请求体", payload)
                
                response = await self.network.post_request(session, url, headers, payload)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.debug_json("CountTokens 响应体", data)
                    try:
                        items = data if isinstance(data, list) else [data]
                        for entry in items:
                            if not isinstance(entry, dict): continue
                            if "errors" in entry:
                                logger.error(f"CountTokens 报错: {entry['errors']}")
                                continue
                            
                            results = entry.get("results", [])
                            for result in results:
                                if "errors" in result:
                                    logger.error(f"CountTokens 报错: {result['errors']}")
                                    continue
                                data_obj = result.get("data", {})
                                ui_data = data_obj.get("ui", {})
                                count_data = ui_data.get("countTokensV2") or data_obj.get("countTokensV2") or data_obj.get("countTokens")
                                if count_data and "totalTokens" in count_data:
                                    return int(count_data["totalTokens"])
                    except Exception as e:
                        logger.error(f"解析 CountTokens 响应失败: {e}")
                else:
                    logger.error(f"CountTokens API 请求失败: {response.status_code}")
                
            return 0
        except Exception as e:
            logger.error(f"远程 Token 计数失败: {e}")
            return 0

    async def count_tokens_remote(self, contents: list[dict[str, Any]], model: str = "gemini-2.5-flash") -> int:
        return await self._count_tokens_with_session(contents, model)


_token_counter = TokenCounter()

async def calculate_usage_metadata(
    prompt_contents: list[dict[str, Any]], 
    response_parts: list[dict[str, Any]],
    request_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """便捷函数：计算完整的 usage metadata"""
    model = "gemini-2.5-flash"
    if request_context and isinstance(request_context, dict):
        downstream = request_context.get("downstream_payload", {})
        if isinstance(downstream, dict):
            model = downstream.get("model", model)
        
    return await _token_counter.calculate_usage_metadata_async(
        prompt_contents, 
        response_parts,
        model
    )
