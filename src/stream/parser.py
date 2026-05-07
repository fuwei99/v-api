

import json
from typing import Any, cast
from src.core.errors import (
    VertexError,
    InternalError,
    parse_error_response,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

def extract_path_index(result: dict[str, Any]) -> int:
    """从 result 对象中提取 path 索引"""
    path = result.get('path', [])
    if not path or not isinstance(path, list):
        return -1
    try:
        
        path_list = cast(list[Any], path)
        for elem in reversed(path_list):
            if isinstance(elem, int):
                return elem
            if isinstance(elem, str) and elem.isdigit():
                return int(elem)
    except (IndexError, ValueError, TypeError):
        pass
    return -1

def clean_json_string(raw_data: str) -> str:
    """清理并规范化 JSON 字符串"""
    cleaned_data = raw_data.strip()
    if cleaned_data.endswith(','):
        cleaned_data = cleaned_data[:-1]
    
    if not cleaned_data.startswith('['):
        cleaned_data = f'[{cleaned_data}]'
    elif not cleaned_data.endswith(']'):
         
         if '}]' not in cleaned_data:
            cleaned_data += ']'
    return cleaned_data

def process_candidate_metadata(candidate_data: dict[str, Any]) -> dict[str, Any]:
    """提取 candidate 级别的元数据（仅提取实际存在的字段）"""
    metadata: dict[str, Any] = {}
    
    finish_reason = candidate_data.get('finishReason')
    if finish_reason:
        metadata['finish_reason'] = finish_reason
        
    if 'finishMessage' in candidate_data:
        metadata['finish_message'] = candidate_data['finishMessage']
    
    
    if candidate_data.get('safetyRatings'):
        metadata['safety_ratings'] = candidate_data['safetyRatings']
        
    if candidate_data.get('citationMetadata'):
        metadata['citation_metadata'] = candidate_data['citationMetadata']
        
    if candidate_data.get('groundingMetadata'):
        metadata['grounding_metadata'] = candidate_data['groundingMetadata']
        
    if 'tokenCount' in candidate_data:
        metadata['token_count'] = candidate_data['tokenCount']
        
    if 'avgLogprobs' in candidate_data:
        metadata['avg_logprobs'] = candidate_data['avgLogprobs']
        
    if 'logprobsResult' in candidate_data:
        metadata['logprobs_result'] = candidate_data['logprobsResult']
    
    
    if candidate_data.get('index') is not None:
        metadata['candidate_index'] = candidate_data['index']
        
    return metadata

def _extract_error_message(item: dict[str, Any]) -> str | None:
    """从单个响应项中提取错误信息（如果有）"""
    
    
    error_obj = item.get('error')
    if error_obj:
        if isinstance(error_obj, dict):
            
            safe_error_obj = cast(dict[str, Any], error_obj)
            return str(safe_error_obj.get('message', str(safe_error_obj)))
        return str(error_obj)

    
    errors = item.get('errors')
    if errors and isinstance(errors, list):
        
        safe_errors = cast(list[Any], errors)
        if safe_errors:
            first_error = safe_errors[0]
            if isinstance(first_error, dict):
                safe_first_error = cast(dict[str, Any], first_error)
                return str(safe_first_error.get('message', str(safe_first_error)))
            return str(first_error)
            
    return None

def _clean_part_fields(part: dict[str, Any]) -> dict[str, Any]:
    """
    清理 part 中的空字段，只保留有实际内容的字段（增强版）
    确保 Part 的纯净度：如果存在工具调用，则不应存在 text 字段。
    """
    cleaned_part: dict[str, Any] = {}
    
    # 优先处理工具调用
    has_tool_call = False
    
    # 1. 处理 functionCall
    func_call = part.get('functionCall')
    if isinstance(func_call, dict):
        fc_dict = cast(dict[str, Any], func_call)
        name = fc_dict.get('name')
        if isinstance(name, str) and name.strip():
            cleaned_part['functionCall'] = {
                "name": name,
                "args": fc_dict.get('args', {})
            }
            has_tool_call = True
            
    # 2. 处理 functionResponse
    if not has_tool_call:
        func_response = part.get('functionResponse')
        if isinstance(func_response, dict):
            name = func_response.get('name')
            if isinstance(name, str) and name.strip():
                cleaned_part['functionResponse'] = func_response
                has_tool_call = True

    # 3. 处理文本 (只有在没有工具调用时才添加，除非文本非空)
    if 'text' in part and part['text'] is not None:
        text_val = part['text']
        if not has_tool_call or (isinstance(text_val, str) and text_val.strip()):
            cleaned_part['text'] = text_val
        
    # 4. 处理 thought / thoughtSignature（尽量保留上游原值）
    if 'thought' in part:
        cleaned_part['thought'] = part['thought']

    if 'thoughtSignature' in part:
        signature_value = part['thoughtSignature']
        if not (isinstance(signature_value, str) and signature_value == "context_engineering_is_the_way_to_go"):
            cleaned_part['thoughtSignature'] = signature_value
        
    # 5. 处理多媒体数据
    if not has_tool_call:
        inline_data = part.get('inlineData')
        if isinstance(inline_data, dict):
            if (isinstance(inline_data.get('data'), str) and inline_data['data'].strip() and
                isinstance(inline_data.get('mimeType'), str) and inline_data['mimeType'].strip()):
                mime_type = str(inline_data['mimeType']).strip()
                data_b64 = str(inline_data['data']).strip()

                cleaned_part['inlineData'] = inline_data

        file_data = part.get('fileData')
        if isinstance(file_data, dict):
            if (isinstance(file_data.get('fileUri'), str) and file_data['fileUri'].strip() and
                isinstance(file_data.get('mimeType'), str) and file_data['mimeType'].strip()):
                cleaned_part['fileData'] = file_data
            
    return cleaned_part

def _merge_content_blocks(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    合并思考块和非思考块的文本内容，同时保持原始顺序
    
    Args:
        parts: 原始的 parts 列表
        
    Returns:
        合并后的 parts 列表
    """
    
    cleaned_parts = [_clean_part_fields(part) for part in parts]
    
    cleaned_parts = [part for part in cleaned_parts if part]
    
    if not cleaned_parts:
        return []

    merged_parts: list[dict[str, Any]] = []
    
    for part in cleaned_parts:
        
        if not merged_parts:
            merged_parts.append(part.copy())
            continue
            
        last_part = merged_parts[-1]
        
        
        can_merge = False
        if 'text' in part and 'text' in last_part:
            
            if part.get('thought', False) == last_part.get('thought', False):
                
                
                other_fields = {'functionCall', 'functionResponse', 'inlineData', 'fileData'}
                if not (set(part.keys()) & other_fields) and not (set(last_part.keys()) & other_fields):
                    can_merge = True
        
        if can_merge:
            last_part['text'] = str(last_part['text']) + str(part['text'])
            
            if part.get('thought', False) and 'thoughtSignature' in part:
                last_part['thoughtSignature'] = part['thoughtSignature']
        else:
            merged_parts.append(part.copy())
            
    return merged_parts

def parse_single_upstream_item(item_dict: dict[str, Any], state: dict[str, Any]) -> None:
    """
    解析单个上游响应项并更新状态。
    
    Args:
        item_dict: 单个 JSON 对象（通常包含 'results' 或 'error'）
        state: 维护解析状态的字典
    """
    
    parsed_error = parse_error_response(item_dict)
    if parsed_error:
        if "Failed to verify action" in parsed_error.message:
            logger.debug(f"忽略预期的认证错误: {parsed_error.message}")
        else:
            state["has_error"] = True
            state["error_message"] = parsed_error.message
            state["error_obj"] = parsed_error
        return

    
    error_msg = _extract_error_message(item_dict)
    if error_msg and not state["has_error"]:
        state["has_error"] = True
        state["error_message"] = error_msg

    
    results = item_dict.get('results', [])
    if not isinstance(results, list):
        return
    
    typed_results: list[dict[str, Any]] = []
    safe_results = cast(list[Any], results)
    for r in safe_results:
        if isinstance(r, dict):
            typed_results.append(cast(dict[str, Any], r))

    
    parsed_error = parse_error_response(typed_results)
    if parsed_error:
        state["has_error"] = True
        state["error_message"] = parsed_error.message
        state["error_obj"] = parsed_error
    
    
    for result in typed_results:
        if result.get('data') is None and 'errors' in result:
            continue

        path_index = extract_path_index(result)
        data = result.get('data')
        
        if isinstance(data, dict):
            _update_state_from_data(state, cast(dict[str, Any], data), path_index)

def parse_upstream_data(raw_data: str) -> dict[str, Any]:
    """
    解析完整的上游原始数据。
    
    Returns:
        包含 parts, finish_reason 和实际存在的元数据的字典
    """
    state: dict[str, Any] = create_initial_state()

    try:
        cleaned_data = clean_json_string(raw_data)
        data_list = json.loads(cleaned_data)
        
        if not isinstance(data_list, list):
            data_list = [data_list]
        
        safe_data_list = cast(list[Any], data_list)

        for item in safe_data_list:
            if isinstance(item, dict):
                parse_single_upstream_item(cast(dict[str, Any], item), state)

    except json.JSONDecodeError as e:
        state["has_error"] = True
        state["error_message"] = f"JSON parse error: {e}"
    except VertexError:
        raise
    except Exception as e:
        logger.error(f"解析过程发生未知错误: {e}")
        state["has_error"] = True
        state["error_message"] = f"Parse error: {str(e)}"
    
    return finalize_state(state)

def create_initial_state() -> dict[str, Any]:
    """创建初始解析状态字典"""
    return {
        "finish_reason": None,
        "finish_message": None,
        "safety_ratings": [],
        "citation_metadata": {},
        "grounding_metadata": {},
        "token_count": None,
        "avg_logprobs": None,
        "logprobs_result": None,
        "candidate_index": 0,
        "prompt_feedback": {},
        "usage_metadata": {},
        "create_time": None,
        "model_version": None,
        "response_id": None,
        "model_status": None,
        "has_error": False,
        "error_message": "",
        "error_obj": None,
        "parts_by_path": {},
        "unindexed_parts": []
    }

def finalize_state(state: dict[str, Any]) -> dict[str, Any]:
    """将中间状态转换为最终结果字典"""
    parts_by_path = cast(dict[int, Any], state['parts_by_path'])
    ordered_parts: list[dict[str, Any]] = [parts_by_path[k] for k in sorted(parts_by_path.keys())]
    unindexed_parts = cast(list[Any], state['unindexed_parts'])
    ordered_parts.extend(unindexed_parts)
    
    final_parts = _merge_content_blocks(ordered_parts)
    
    result: dict[str, Any] = {
        "parts": final_parts
    }
    
    excluded_keys = ['parts_by_path', 'unindexed_parts']
    result.update({k: v for k, v in state.items() if k not in excluded_keys})
    return result

def _update_state_from_data(state: dict[str, Any], data: dict[str, Any], path_index: int):
    """从数据对象更新解析状态（仅提取实际存在的字段）"""
    
    if data.get('promptFeedback'):
        state['prompt_feedback'] = data['promptFeedback']
    if 'usageMetadata' in data:
        state['usage_metadata'] = data['usageMetadata']
    if 'createTime' in data:
        state['create_time'] = data['createTime']
    if 'modelVersion' in data:
        state['model_version'] = data['modelVersion']
    if 'responseId' in data:
        state['response_id'] = data['responseId']
    if 'modelStatus' in data:
        state['model_status'] = data['modelStatus']
            
    
    candidates = data.get('candidates', [])
    for candidate in candidates:
        
        meta = process_candidate_metadata(candidate)
        for k, v in meta.items():
            if v is not None and v != [] and v != {}:
                state[k] = v

        
        content = candidate.get('content', {})
        parts = content.get('parts', [])
        for part in parts:
            if path_index != -1:
                state['parts_by_path'][path_index] = part
            else:
                state['unindexed_parts'].append(part)
