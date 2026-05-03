

import json
import time
import random
from typing import Any, cast

from src.core.config import load_config
from src.core.errors import (
    VertexError,
    InternalError,
    parse_error_response,
)
from src.api.model_config import ModelConfigBuilder
from src.utils.logger import get_logger
from src.utils.string_utils import snake_to_camel, camel_to_snake

logger = get_logger(__name__)

class RequestTransformer:
    """
    请求参数转换器，负责将标准 Gemini API 格式的请求体转换为 Vertex AI 匿名接口所需的 GraphQL 格式。
    
    核心功能：
    1. 结构对齐：处理 contents, tools, safetySettings 等字段。
    2. 格式修复：自动处理 base64 填充、合并连续相同角色的对话块。
    3. 特殊逻辑：注入“防标记”混淆文本，处理 systemInstruction 向后兼容。
    4. Schema 转换：将 JSON Schema 转换为 Vertex 内部的 Key-Value 数组格式。
    """
    
    def __init__(self, model_builder: ModelConfigBuilder):
        self.model_builder = model_builder
        self.config = load_config()

    def build_vertex_payload(
        self,
        model: str,
        gemini_payload: dict[str, Any],
        original_body: dict[str, Any],
        kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """
        核心逻辑：将 Gemini 格式的请求体（由用户或下游发送）组装成 Vertex AI 匿名 GraphQL 接口接受的 payload。
        
        参数:
            model: 目标模型 ID。
            gemini_payload: 符合 Gemini API 标准的原始请求内容（contents, tools 等）。
            original_body: 用于提供 querySignature 和 operationName 的原始结构。
            kwargs: 额外的生成配置参数。
            
        流程：
        1. 初始化 variables，映射后端模型名称。
        2. 使用 Pydantic (GeminiPayload) 验证并清理输入字段，确保符合标准结构。
        3. 注入防标记文本（Anti-Tracking），防止请求被上游标记。
        4. 处理 contents：包括修复 Base64 格式、合并连续角色、过滤空内容等复杂处理。
        5. 工具处理：标准化 tools 格式，并将 JSON Schema 递归转换为后端特定的 native 格式。
        6. 生成配置：合并全局配置与用户配置。
        7. 组装最终的 GraphQL body 结构。
        """
        original_vars: Any = original_body.get('variables', {})
        new_variables: dict[str, Any]
        if hasattr(original_vars, 'model_dump'):
             new_variables = cast(dict[str, Any], original_vars.model_dump())
        elif isinstance(original_vars, dict):
            new_variables = {str(k): v for k, v in cast(dict[Any, Any], original_vars).items()}
        else:
             new_variables = {}

        target_model = self.model_builder.parse_model_name(model)
        new_variables['model'] = target_model

        
        supported_fields =[
            'contents', 'tools', 'toolConfig', 'systemInstruction',
            'safetySettings', 'generationConfig'
        ]
        
        try:
            from src.core.types import GeminiPayload
            gemini_payload_obj = GeminiPayload.model_validate(gemini_payload)
            dumped_payload = gemini_payload_obj.model_dump(by_alias=True, exclude_none=True)
            
            for field in supported_fields:
                if field in dumped_payload:
                     new_variables[field] = dumped_payload[field]
        except Exception as e:
            logger.debug(f"Pydantic 验证失败，使用基础转换: {e}")
            
            for field in supported_fields:
                
                if field in gemini_payload:
                    new_variables[field] = gemini_payload[field]
                else:
                    
                    snake_field = camel_to_snake(field)
                    if snake_field in gemini_payload:
                        new_variables[field] = gemini_payload[snake_field]

        
        if self.config.get("anti_tracking", False):
            self._inject_anti_tracking(new_variables)

        
        self._handle_system_instruction(new_variables)

        
        if 'contents' in new_variables:
            converted_contents = self._handle_inline_data_case(new_variables['contents'])
            converted_contents = self._handle_base64_in_contents(converted_contents)
            
            converted_contents = self._merge_contiguous_roles(converted_contents)
            
            converted_contents = self._filter_empty_contents(converted_contents)
            
            converted_contents = self._handle_thought_signature(converted_contents)
            new_variables['contents'] = converted_contents
        
        
        if 'tools' in new_variables:
            normalized_tools = self._normalize_tools_format(new_variables['tools'])
            if normalized_tools:
                new_variables['tools'] = normalized_tools
            else:
                
                del new_variables['tools']
                if 'toolConfig' in new_variables:
                    del new_variables['toolConfig']
        
        
        if 'toolConfig' in new_variables:
            new_variables['toolConfig'] = self._convert_tools_format(new_variables['toolConfig'])

        
        gen_config = self.model_builder.build_generation_config(
            gen_config={},
            gemini_payload=gemini_payload,
            **kwargs
        )
        if gen_config:
            
            if 'logitBias' in gen_config and isinstance(gen_config['logitBias'], dict):
                bias_dict = gen_config.pop('logitBias')
                gen_config['logitBias'] = [{"key": str(k), "value": v} for k, v in bias_dict.items()]

            
            if 'responseSchema' in gen_config and isinstance(gen_config['responseSchema'], dict):
                gen_config['responseSchema'] = self._to_native_schema(gen_config['responseSchema'])
                
            new_variables['generationConfig'] = gen_config
            
        
        if 'safetySettings' not in new_variables and 'safety_settings' not in gemini_payload:
            new_variables['safetySettings'] = self.model_builder.build_safety_settings()

        new_body: dict[str, Any] = {
            "querySignature": original_body.get('querySignature'),
            "operationName": original_body.get('operationName'),
            "variables": new_variables
        }
        
        return new_body

    def _inject_anti_tracking(self, new_variables: dict[str, Any]) -> None:
        """注入防标记内容到系统指令最上方"""
        letters = "abcdefghijklmnopq"
        def r_let() -> str: return random.choice(letters)
        def r_num() -> str: return str(random.randint(1, 999999))
        
        parts =[
            r_let(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(),
            r_num(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(),
            r_num(), r_let(), r_num(), r_let(), r_num(), r_num(), r_let(),
            r_let(), r_num(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(),
            r_num(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(),
            r_num(), r_let(), r_num(), r_let(), r_num(), r_num(), r_let(),
            r_let(), r_num(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(),
            r_num(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(),
            r_num(), r_let(), r_num(), r_let(), r_num(), r_num(), r_let(),
            r_let(), r_num(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(),
            r_num(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(),
            r_num(), r_let(), r_num(), r_let(), r_num(), r_num(), r_let(),
            r_let(), r_num(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(),
            r_num(), r_num(), r_let(), r_let(), r_num(), r_let(), r_let(),
            r_num(), r_let(), r_num(), r_let(), r_num(), r_num(), r_let(),
            r_let(), r_num(), r_num(), r_let()
        ]
        random_str = "".join(parts)
        anti_tracking_prefix = f"<|no-trans|>meaningless test: {random_str}\n\n[遵循如下指令]\n\n"
        
        sys_inst = new_variables.get('systemInstruction')
        if sys_inst:
            if isinstance(sys_inst, dict) and 'parts' in sys_inst and isinstance(sys_inst['parts'], list):
                parts_list = cast(list[Any], sys_inst['parts'])
                if len(parts_list) > 0 and isinstance(parts_list[0], dict) and 'text' in parts_list[0]:
                    parts_list[0]['text'] = anti_tracking_prefix + str(parts_list[0]['text'])
                else:
                    parts_list.insert(0, {'text': anti_tracking_prefix})
            elif isinstance(sys_inst, str):
                new_variables['systemInstruction'] = {'parts': [{'text': anti_tracking_prefix + sys_inst}]}
        else:
            new_variables['systemInstruction'] = {
                'parts': [{'text': anti_tracking_prefix}]
            }

    def _convert_tools_format(self, data: Any) -> Any:
        """
        根据逆向出的 CLq 注册表，精准处理工具格式。
        - JsonObject (args, response) 保持原样。
        - Map/Schema (parameters) 执行 Key/Value 数组转换。
        """
        if isinstance(data, dict):
            new_dict: dict[str, Any] = {}
            data_dict: dict[str, Any] = cast(dict[str, Any], data)
            for k, v in data_dict.items():
                camel_key = snake_to_camel(k)
                
                
                if camel_key in ["parameters", "parametersJsonSchema", "parameters_json_schema"]:
                    if isinstance(v, dict):
                        new_dict["parameters"] = self._to_native_schema(cast(dict[str, Any], v))
                    continue

                
                if camel_key in ["args", "response"]:
                    new_dict[camel_key] = v
                    continue

                
                if isinstance(v, (dict, list)):
                    new_dict[camel_key] = self._convert_tools_format(v)
                else:
                    new_dict[camel_key] = v
            return new_dict
        elif isinstance(data, list):
            return [self._convert_tools_format(item) for item in cast(list[Any], data)]
        else:
            return data

    def _to_native_schema(self, schema: Any) -> Any:
        """
        全递归 Schema 转换器 (根据最新逆向代码修正)
        处理数值转字符串、类型大写化、处理数组型 Type 以及字段清理。
        """
        if not isinstance(schema, dict):
            return schema

        native_schema = schema.copy()

        
        if 'type' in native_schema:
            t_raw = native_schema['type']
            target_type = "STRING" 

            
            if isinstance(t_raw, list):
                valid_types = [str(x).upper() for x in t_raw if str(x).lower() != "null"]
                if valid_types:
                    target_type = valid_types[0]
            elif isinstance(t_raw, str):
                target_type = t_raw.upper()

            
            if target_type in ["INTEGER", "NUMBER", "BOOLEAN", "ARRAY", "OBJECT", "STRING"]:
                native_schema['type'] = target_type
            else:
                native_schema['type'] = "STRING" 

        
        
        numeric_constraints = [
            'minItems', 'maxItems', 
            'minProperties', 'maxProperties', 
            'minLength', 'maxLength'
        ]
        for field in numeric_constraints:
            if field in native_schema and native_schema[field] is not None:
                native_schema[field] = str(native_schema[field])

        
        if 'properties' in native_schema and isinstance(native_schema['properties'], dict):
            props_dict = native_schema.pop('properties')
            native_schema['properties'] = [
                {"key": str(k), "value": self._to_native_schema(v)}
                for k, v in props_dict.items()
            ]

        
        if 'defs' in native_schema and isinstance(native_schema['defs'], dict):
            defs_dict = native_schema.pop('defs')
            native_schema['defs'] = [
                {"key": str(k), "value": self._to_native_schema(v)}
                for k, v in defs_dict.items()
            ]

        
        if 'items' in native_schema and isinstance(native_schema['items'], dict):
            native_schema['items'] = self._to_native_schema(native_schema['items'])

        for list_key in ['anyOf', 'oneOf', 'allOf']:
            if list_key in native_schema and isinstance(native_schema[list_key], list):
                native_schema[list_key] = [self._to_native_schema(item) for item in native_schema[list_key]]

        
        native_schema.pop('additionalPropertiesSchema', None)
        native_schema.pop('prefixItems', None)
        
        native_schema.pop('additionalProperties', None) 

        return native_schema

    def _merge_contiguous_roles(self, contents: Any) -> Any:
        """合并连续的相同角色的 content 块"""
        if not isinstance(contents, list):
            return contents

        merged_contents: list[Any] = []
        for content in cast(list[Any], contents):
            if not isinstance(content, dict):
                merged_contents.append(content)
                continue

            content_dict = cast(dict[str, Any], content)
            role = content_dict.get('role')
            
            if merged_contents and isinstance(merged_contents[-1], dict) and merged_contents[-1].get('role') == role and role is not None:
                last_content = merged_contents[-1]
                last_parts = last_content.get('parts', [])
                curr_parts = content_dict.get('parts', [])
                
                if isinstance(last_parts, list) and isinstance(curr_parts, list):
                    new_content = last_content.copy()
                    new_content['parts'] = list(last_parts) + list(curr_parts)
                    merged_contents[-1] = new_content
                else:
                    merged_contents.append(content_dict)
            else:
                merged_contents.append(content_dict)
                
        return merged_contents
    
    def _handle_system_instruction(self, new_variables: dict[str, Any]) -> None:
        """处理 systemInstruction：如果没有 user content，则转换为 user message"""
        system_instruction_content = new_variables.get('systemInstruction')
        if not system_instruction_content:
            return
            
        contents = new_variables.get('contents',[])
        
        
        contents_list: list[Any] = cast(list[Any], contents) if isinstance(contents, list) else[]
        has_user_role = any(
            isinstance(content, dict) and cast(dict[str, Any], content).get('role') == 'user'
            for content in contents_list
        )
        
        if has_user_role:
            return
            
        
        text_from_system = self._extract_text_from_instruction(system_instruction_content)
        if not text_from_system:
            return
            
        
        user_message = {
            'role': 'user',
            'parts': [{'text': text_from_system}]
        }
        
        
        new_contents: list[Any] = list(contents_list)
        new_contents.insert(0, user_message)
        new_variables['contents'] = new_contents
        del new_variables['systemInstruction']
    
    def _extract_text_from_instruction(self, instruction: Any) -> str:
        """从 system instruction 中提取文本内容"""
        if isinstance(instruction, str):
            return instruction
        elif isinstance(instruction, dict):
            instruction_dict = cast(dict[str, Any], instruction)
            parts = instruction_dict.get('parts',[])
            if isinstance(parts, list):
                text_parts =[]
                for part in parts:
                    if isinstance(part, dict) and 'text' in part:
                        text_parts.append(str(part['text']))
                return "".join(text_parts)
        return ""
    
    def _normalize_tools_format(self, tools: Any) -> list[dict[str, Any]]:
        """标准化 tools 格式为 Vertex AI 期望的格式 (List[Tool])"""
        converted_tools: Any = self._convert_tools_format(tools)
        
        if isinstance(converted_tools, dict):
            
            if 'functionDeclarations' in converted_tools:
                return[cast(dict[str, Any], converted_tools)]
            
            if 'name' in converted_tools:
                return [{"functionDeclarations":[cast(dict[str, Any], converted_tools)]}]
            return[]
            
        if not isinstance(converted_tools, list) or len(cast(list[Any], converted_tools)) == 0:
            return[]
            
        converted_tools_list: list[Any] = cast(list[Any], converted_tools)
        first_item: Any = converted_tools_list[0]
        if not isinstance(first_item, dict):
            return[]
            
        
        if 'name' in first_item and 'functionDeclarations' not in first_item:
            return [{"functionDeclarations": cast(list[dict[str, Any]], converted_tools_list)}]
            
        
        
        if 'functionDeclarations' in first_item:
            return cast(list[dict[str, Any]], converted_tools_list)
            
        return[]

    def _handle_inline_data_case(self, contents: Any) -> Any:
        """
        递归处理内容，保护工具调用参数名并对齐 ID 字段
        """
        if isinstance(contents, list):
            return [self._handle_inline_data_case(item) for item in cast(list[Any], contents)]
        if isinstance(contents, dict):
            new_dict: dict[str, Any] = {}
            for k, v in cast(dict[str, Any], contents).items():
                camel_k = snake_to_camel(k)
                if camel_k == 'inlineData' and isinstance(v, dict):
                    v_dict = cast(dict[str, Any], v)
                    new_inline_data = {}
                    for ik, iv in v_dict.items():
                        camel_ik = snake_to_camel(ik)
                        new_inline_data[camel_ik] = iv
                    new_dict['inlineData'] = new_inline_data
                elif camel_k == 'functionCall' and isinstance(v, dict):
                    v_dict = cast(dict[str, Any], v)
                    new_fc = {}
                    
                    fid = v_dict.get('id') or v_dict.get('tool_call_id') or v_dict.get('toolCallId')
                    if fid: new_fc['id'] = fid
                    for ik, iv in v_dict.items():
                        cik = snake_to_camel(ik)
                        if cik == 'args': new_fc[cik] = iv 
                        elif cik not in ['id', 'toolCallId']: new_fc[cik] = self._handle_inline_data_case(iv)
                    new_dict['functionCall'] = new_fc
                elif camel_k == 'functionResponse' and isinstance(v, dict):
                    v_dict = cast(dict[str, Any], v)
                    new_fr = {}
                    fid = v_dict.get('id') or v_dict.get('tool_call_id') or v_dict.get('toolCallId')
                    if fid: new_fr['id'] = fid
                    for ik, iv in v_dict.items():
                        cik = snake_to_camel(ik)
                        if cik == 'response': new_fr[cik] = iv 
                        elif cik not in ['id', 'toolCallId']: new_fr[cik] = self._handle_inline_data_case(iv)
                    new_dict['functionResponse'] = new_fr
                else:
                    new_dict[camel_k] = self._handle_inline_data_case(v)
            return new_dict
        return contents
        
    def _handle_base64_in_contents(self, contents: Any) -> Any:
        """
        递归处理 contents 中的 base64 数据。
        将 URL-safe Base64 转换为标准 Base64 并补全 padding。
        """
        try:
            if isinstance(contents, list):
                res_list: list[Any] =[self._handle_base64_in_contents(item) for item in cast(list[Any], contents)]
                return cast(Any, res_list)
            if isinstance(contents, dict):
                new_dict: dict[str, Any] = {}
                for k, v in cast(dict[str, Any], contents).items():
                    if k == 'inlineData' and isinstance(v, dict):
                        v_dict = cast(dict[str, Any], v)
                        if 'data' in v_dict and isinstance(v_dict['data'], str):
                            try:
                                b64_data: str = v_dict['data']
                                b64_data = b64_data.replace('-', '+').replace('_', '/')
                                padding = len(b64_data) % 4
                                if padding:
                                    b64_data += '=' * (4 - padding)
                                
                                new_inline_data = v_dict.copy()
                                new_inline_data['data'] = b64_data
                                new_dict[k] = new_inline_data
                                continue
                            except Exception:
                                pass
                    new_dict[k] = self._handle_base64_in_contents(v)
                return cast(Any, new_dict)
            return contents
        except Exception as e:
            logger.warning(f"Base64 内容处理失败: {e}")
            return cast(Any, contents)

    def _filter_empty_contents(self, contents: Any) -> Any:
        """
        过滤掉空的 contents（parts 为空数组的消息）
        Vertex AI 要求每个 content 至少包含一个 part
        """
        if not isinstance(contents, list):
            return contents

        filtered_contents: list[Any] = []
        contents_list: list[Any] = cast(list[Any], contents)

        
        last_model_function_calls: list[str] = []
        
        call_id_map: dict[str, str] = {} 
        
        response_index_in_content = 0

        for content in contents_list:
            if not isinstance(content, dict):
                continue
            content_dict = cast(dict[str, Any], content)
            role = content_dict.get('role')
            parts = content_dict.get('parts', [])

            if role == 'model':
                
                last_model_function_calls = []
                if isinstance(parts, list):
                    for part in cast(list[Any], parts):
                        if isinstance(part, dict):
                            func_call = part.get('functionCall')
                            if isinstance(func_call, dict) and func_call.get('name'):
                                name = str(func_call['name'])
                                fid = func_call.get('id')
                                last_model_function_calls.append(name)
                                if fid:
                                    call_id_map[str(fid)] = name
            
            elif role == 'function':
                
                response_index_in_content = 0

            if isinstance(parts, list) and len(cast(list[Any], parts)) > 0:
                parts_list: list[Any] = cast(list[Any], parts)
                valid_parts: list[Any] = []
                for part in parts_list:
                    if isinstance(part, dict):
                        part_dict = cast(dict[str, Any], part)

                        
                        is_func_response = 'functionResponse' in part_dict
                        
                        
                        
                        cleaned_part = self._clean_part_metadata(
                            part_dict, 
                            last_model_function_calls,
                            response_index_in_content if is_func_response else -1,
                            call_id_map
                        )
                        
                        if is_func_response:
                            response_index_in_content += 1
                            
                        if cleaned_part:
                            valid_parts.append(cleaned_part)

                if valid_parts:
                    filtered_content = content_dict.copy()
                    filtered_content['parts'] = valid_parts
                    filtered_contents.append(filtered_content)
                else:
                    logger.debug(f"过滤掉空的 content: role={content_dict.get('role', 'unknown')}")
            else:
                
                if not isinstance(content, dict):
                    filtered_contents.append(content)
                else:
                    logger.debug(f"过滤掉空的 content: role={content_dict.get('role', 'unknown')}")

        return filtered_contents

    def _clean_part_metadata(
        self, 
        part_dict: dict[str, Any], 
        function_call_names: list[str],
        response_index: int = -1,
        call_id_map: dict[str, str] | None = None
    ) -> dict[str, Any] | None:
        """
        清理 part 中的空元数据字段，修复无效的 functionResponse
        
        Args:
            part_dict: 原始 part 字典
            function_call_names: 前序 model 消息中的函数调用名称列表
            response_index: 当前 functionResponse 在消息中的位置索引
            
        Returns:
            清理后的 part 字典，如果 part 无效则返回 None
        """
        cleaned_part: dict[str, Any] = {}
        has_valid_content = False
        
        
        if 'text' in part_dict:
            text_value = part_dict['text']
            if text_value is not None:
                cleaned_part['text'] = text_value
                has_valid_content = True
        
        
        if 'thought' in part_dict:
            cleaned_part['thought'] = part_dict['thought']
        
        
        if 'thoughtSignature' in part_dict:
            cleaned_part['thoughtSignature'] = part_dict['thoughtSignature']
        
        
        if 'functionCall' in part_dict:
            func_call = part_dict['functionCall']
            if isinstance(func_call, dict):
                func_call_dict = cast(dict[str, Any], func_call)
                if func_call_dict.get('name'):  
                    cleaned_part['functionCall'] = func_call
                    has_valid_content = True
        
        
        if 'functionResponse' in part_dict:
            func_response = part_dict['functionResponse']
            if isinstance(func_response, dict):
                fixed_resp = cast(dict[str, Any], func_response).copy()
                current_name = fixed_resp.get('name')
                current_id = fixed_resp.get('id')
                
                
                
                if not current_name and current_id and call_id_map:
                    current_name = call_id_map.get(str(current_id))
                    if current_name:
                        fixed_resp['name'] = current_name
                
                
                if not current_name and 0 <= response_index < len(function_call_names):
                    current_name = function_call_names[response_index]
                    fixed_resp['name'] = current_name

                
                if current_name:
                    cleaned_part['functionResponse'] = fixed_resp
                    has_valid_content = True
                else:
                    logger.debug("丢弃缺失工具名且无法推断的 functionResponse")
        
        
        if 'inlineData' in part_dict:
            inline_data = part_dict['inlineData']
            if isinstance(inline_data, dict):
                inline_data_dict = cast(dict[str, Any], inline_data)
                
                if (inline_data_dict.get('data') and
                    str(inline_data_dict['data']).strip() and
                    inline_data_dict.get('mimeType') and
                    str(inline_data_dict['mimeType']).strip()):
                    cleaned_part['inlineData'] = inline_data
                    has_valid_content = True
        
        
        if 'fileData' in part_dict:
            file_data = part_dict['fileData']
            if isinstance(file_data, dict):
                file_data_dict = cast(dict[str, Any], file_data)
                
                if (file_data_dict.get('fileUri') and
                    str(file_data_dict['fileUri']).strip() and
                    file_data_dict.get('mimeType') and
                    str(file_data_dict['mimeType']).strip()):
                    cleaned_part['fileData'] = file_data
                    has_valid_content = True
        
        
        if has_valid_content:
            return cleaned_part
        else:
            logger.debug("过滤掉没有有效内容的 part")
            return None

    def _handle_thought_signature(self, contents: Any) -> Any:
        """
        处理 thoughtSignature 字段
        将 GCP 标准 API 的 context_engineering_is_the_way_to_go 替换为 Vertex AI 匿名端点所需的 skip_thought_signature_validator
        """
        if isinstance(contents, list):
            return[self._handle_thought_signature(item) for item in cast(list[Any], contents)]
        if isinstance(contents, dict):
            new_dict: dict[str, Any] = {}
            contents_dict: dict[str, Any] = cast(dict[str, Any], contents)
            for k, v in contents_dict.items():
                if k == 'parts' and isinstance(v, list):
                    v_list: list[Any] = cast(list[Any], v)
                    new_parts: list[Any] = []
                    for part in v_list:
                        if isinstance(part, dict):
                            new_part: dict[str, Any] = cast(dict[str, Any], part).copy()
                            if 'thoughtSignature' in new_part:
                                signature_value = new_part['thoughtSignature']
                                if isinstance(signature_value, str):
                                    if signature_value == "context_engineering_is_the_way_to_go":
                                        new_part['thoughtSignature'] = "skip_thought_signature_validator"
                            new_parts.append(new_part)
                        else:
                            new_parts.append(part)
                    new_dict[k] = new_parts
                else:
                    new_dict[k] = self._handle_thought_signature(v) if isinstance(v, (dict, list)) else v
            return new_dict
        return contents

    @staticmethod
    def prepare_headers(creds: dict[str, Any]) -> dict[str, str]:
        """准备请求头"""
        
        headers = RequestTransformer._extract_headers_from_creds(creds)
        
        
        headers['content-type'] = 'application/json'
        
        
        problematic_headers =[
            'content-length', 'Content-Length', 'host', 'Host',
            'connection', 'Connection', 'accept-encoding'
        ]
        for header in problematic_headers:
            headers.pop(header, None)
            
        return headers
    
    @staticmethod
    def _extract_headers_from_creds(creds: dict[str, Any]) -> dict[str, str]:
        """从凭证中提取头信息"""
        if hasattr(creds, 'model_dump') and hasattr(creds, 'headers'):
            headers_attr = getattr(creds, 'headers')
            if isinstance(headers_attr, dict):
                return cast(dict[str, str], headers_attr).copy()
        
        raw_headers = creds.get('headers')
        if isinstance(raw_headers, dict):
            return cast(dict[str, str], raw_headers).copy()
            
        return {}


class ResponseAggregator:
    """响应聚合器"""

    @staticmethod
    async def aggregate_stream(stream_generator: Any, _raw_image_response: bool = False) -> dict[str, Any]:
        """
        聚合流式响应为非流式对象
        """
        all_parts: list[dict[str, Any]] =[]
        finish_reason: str | None = None
        finish_message: str | None = None
        safety_ratings: list[dict[str, Any]] =[]
        citation_metadata: dict[str, Any] = {}
        grounding_metadata: dict[str, Any] = {}
        token_count: int | None = None
        avg_logprobs: float | None = None
        logprobs_result: dict[str, Any] | None = None
        candidate_index = 0
        usage_metadata: dict[str, Any] = {}
        
        create_time: str | None = None
        model_version: str | None = None
        prompt_feedback: dict[str, Any] = {}
        response_id: str | None = None
        model_status: dict[str, Any] | None = None
        
        try:
            async for chunk_str in stream_generator:
                actual_json_str = chunk_str.strip()
                if actual_json_str.startswith("data: "):
                    actual_json_str = actual_json_str[6:]
                
                if not actual_json_str:
                    continue
                
                try:
                    chunk = json.loads(actual_json_str)
                    
                    
                    parsed_error = parse_error_response(chunk)
                    if parsed_error:
                        raise parsed_error
                    
                    
                    create_time = chunk.get('createTime') or create_time
                    model_version = chunk.get('modelVersion') or model_version
                    prompt_feedback = chunk.get('promptFeedback', {}) or prompt_feedback
                    response_id = chunk.get('responseId') or response_id
                    usage_metadata = chunk.get('usageMetadata', {}) or usage_metadata
                    model_status = chunk.get('modelStatus') or model_status
                    
                    candidates = chunk.get('candidates', [])
                    if candidates:
                        candidate = candidates[0]
                        
                        
                        content_obj = candidate.get('content', {})
                        parts = content_obj.get('parts',[])
                        if parts:
                            all_parts.extend(parts)
                        
                        
                        finish_reason = candidate.get('finishReason') or finish_reason
                        finish_message = candidate.get('finishMessage') or finish_message
                        safety_ratings = candidate.get('safetyRatings') or safety_ratings
                        citation_metadata = candidate.get('citationMetadata') or citation_metadata
                        grounding_metadata = candidate.get('groundingMetadata') or grounding_metadata
                        if candidate.get('tokenCount') is not None:
                            token_count = candidate['tokenCount']
                        if candidate.get('avgLogprobs') is not None:
                            avg_logprobs = candidate['avgLogprobs']
                        if candidate.get('logprobsResult') is not None:
                            logprobs_result = candidate['logprobsResult']
                        if candidate.get('index') is not None:
                            candidate_index = candidate['index']
                            
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON 解析失败，跳过此块: {e}")
                    continue
                    
        except VertexError:
            raise
        except Exception as e:
            raise InternalError(message=f"Non-streaming request error: {e}")
        
        
        full_text_content = "".join(str(p['text']) for p in all_parts if 'text' in p)

        if full_text_content.startswith("![Generated Image](data:"):
            start_idx = full_text_content.find('(') + 1
            end_idx = full_text_content.rfind(')')
            data_url = full_text_content[start_idx:end_idx]
            if _raw_image_response:
                try:
                    _, encoded = data_url.split(',', 1)
                    return {"created": int(time.time()), "data": [{"b64_json": encoded}]}
                except Exception:
                    return {"created": int(time.time()), "data":[]}
            else:
                return {"resultUrl": data_url}
        
        
        if not all_parts:
            all_parts =[{"text": " "}]
        
        result_candidate: dict[str, Any] = {
            "index": candidate_index
        }
        if finish_reason:
            result_candidate["finishReason"] = finish_reason.upper()
            
        result_candidate["content"] = {
            "parts": all_parts,
            "role": "model"
        }
        
        
        optional_candidate_fields: dict[str, Any] = {
            "finishMessage": finish_message,
            "safetyRatings": safety_ratings,
            "citationMetadata": citation_metadata,
            "groundingMetadata": grounding_metadata,
            "tokenCount": token_count,
            "avgLogprobs": avg_logprobs,
            "logprobsResult": logprobs_result
        }
        
        for key, value in optional_candidate_fields.items():
            if value is not None and value != [] and value != {}:
                result_candidate[key] = value
        
        
        result: dict[str, Any] = {"candidates":[result_candidate]}
        
        optional_result_fields: dict[str, Any] = {
            "createTime": create_time,
            "modelVersion": model_version,
            "promptFeedback": prompt_feedback,
            "responseId": response_id,
            "usageMetadata": usage_metadata,
            "modelStatus": model_status
        }
        
        for key, value in optional_result_fields.items():
            if value is not None and value != {} and value != "":
                result[key] = value
        
        return result