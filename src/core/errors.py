

import json
from typing import Any
from enum import Enum

class ErrorStatus(str, Enum):
    """Vertex AI API 错误状态码 (基于 gRPC 标准)"""
    OK = "OK"                                   
    CANCELLED = "CANCELLED"                     
    UNKNOWN = "UNKNOWN"                         
    INVALID_ARGUMENT = "INVALID_ARGUMENT"       
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"     
    NOT_FOUND = "NOT_FOUND"                     
    ALREADY_EXISTS = "ALREADY_EXISTS"           
    PERMISSION_DENIED = "PERMISSION_DENIED"     
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"   
    FAILED_PRECONDITION = "FAILED_PRECONDITION" 
    ABORTED = "ABORTED"                         
    OUT_OF_RANGE = "OUT_OF_RANGE"               
    UNIMPLEMENTED = "UNIMPLEMENTED"             
    INTERNAL = "INTERNAL"                       
    UNAVAILABLE = "UNAVAILABLE"                 
    DATA_LOSS = "DATA_LOSS"                     
    UNAUTHENTICATED = "UNAUTHENTICATED"         


GRPC_TO_HTTP: dict[ErrorStatus, int] = {
    ErrorStatus.OK: 200,
    ErrorStatus.CANCELLED: 499,
    ErrorStatus.UNKNOWN: 500,
    ErrorStatus.INVALID_ARGUMENT: 400,
    ErrorStatus.DEADLINE_EXCEEDED: 504,
    ErrorStatus.NOT_FOUND: 404,
    ErrorStatus.ALREADY_EXISTS: 409,
    ErrorStatus.PERMISSION_DENIED: 403,
    ErrorStatus.RESOURCE_EXHAUSTED: 429,
    ErrorStatus.FAILED_PRECONDITION: 400,
    ErrorStatus.ABORTED: 409,
    ErrorStatus.OUT_OF_RANGE: 400,
    ErrorStatus.UNIMPLEMENTED: 501,
    ErrorStatus.INTERNAL: 500,
    ErrorStatus.UNAVAILABLE: 503,
    ErrorStatus.DATA_LOSS: 500,
    ErrorStatus.UNAUTHENTICATED: 401,
}

class VertexError(Exception):
    """
    Vertex AI 代理错误基类。
    
    该类实现了将上游异常（如 gRPC 状态码）映射为标准 HTTP 状态码，
    并提供了便捷的方法将错误转换为 Gemini API 兼容的 JSON 或 SSE 响应格式。
    """
    
    def __init__(
        self,
        message: str,
        code: int | None = None,
        status: str | ErrorStatus | None = None,
        details: dict[str, Any] | None = None,
        upstream_response: str | None = None
    ):
        self.message = message
        
        
        if isinstance(status, ErrorStatus):
            self.status = status.value
        elif isinstance(status, str):
            try:
                self.status = ErrorStatus(status).value
            except ValueError:
                self.status = ErrorStatus.UNKNOWN.value
        else:
            self.status = ErrorStatus.UNKNOWN.value
            
        
        if code is not None:
            self.code = code
        else:
            self.code = GRPC_TO_HTTP.get(ErrorStatus(self.status), 500)
            
        self.details = details or {}
        self.upstream_response = upstream_response
        super().__init__(message)
    
    def to_Dict(self) -> dict[str, Any]:
        """转换为 Gemini API 兼容的错误响应格式"""
        error_dict: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
                "status": self.status
            }
        }
        if self.details:
            error_dict["error"]["details"] = self.details
        return error_dict
    
    def to_json(self) -> str:
        return json.dumps(self.to_Dict(), ensure_ascii=False)
    
    def to_sse(self) -> str:
        return f"data: {self.to_json()}\n\n"
    
    @property
    def is_retryable(self) -> bool:
        """判断此错误是否可重试"""
        
        if self.code in {408, 429, 500, 502, 503, 504}:
            return True
        
        if isinstance(self, AuthenticationError):
            return True
        return False

class ClientError(VertexError):
    """客户端错误 (4xx)"""
    pass

class ServerError(VertexError):
    """服务端错误 (5xx)"""
    pass

class AuthenticationError(ClientError):
    """认证错误 (401/403)"""
    def __init__(self, message: str = "Authentication failed", details: dict[str, Any] | None = None, upstream_response: str | None = None):
        super().__init__(message, 401, ErrorStatus.UNAUTHENTICATED, details, upstream_response)

class PermissionDeniedError(ClientError):
    """权限拒绝错误 (403)"""
    def __init__(self, message: str = "Permission denied", details: dict[str, Any] | None = None, upstream_response: str | None = None):
        super().__init__(message, 403, ErrorStatus.PERMISSION_DENIED, details, upstream_response)

class InvalidArgumentError(ClientError):
    """参数错误 (400)"""
    def __init__(self, message: str = "Invalid argument", details: dict[str, Any] | None = None, upstream_response: str | None = None):
        super().__init__(message, 400, ErrorStatus.INVALID_ARGUMENT, details, upstream_response)

class NotFoundError(ClientError):
    """资源不存在错误 (404)"""
    def __init__(self, message: str = "Resource not found", details: dict[str, Any] | None = None, upstream_response: str | None = None):
        super().__init__(message, 404, ErrorStatus.NOT_FOUND, details, upstream_response)

class RateLimitError(ClientError):
    """速率限制/资源耗尽错误 (429)"""
    def __init__(self, message: str = "Resource exhausted", details: dict[str, Any] | None = None, retry_after: int | None = None, upstream_response: str | None = None):
        super().__init__(message, 429, ErrorStatus.RESOURCE_EXHAUSTED, details, upstream_response)
        self.retry_after = retry_after

class InternalError(ServerError):
    """内部服务器错误 (500)"""
    def __init__(self, message: str = "Internal server error", details: dict[str, Any] | None = None, upstream_response: str | None = None):
        super().__init__(message, 500, ErrorStatus.INTERNAL, details, upstream_response)

class EmptyResponseError(ServerError):
    """上游返回空响应"""
    def __init__(self, message: str = "Upstream returned empty response", details: dict[str, Any] | None = None, upstream_response: str | None = None):
        super().__init__(message, 502, ErrorStatus.INTERNAL, details, upstream_response)

class UpstreamError(ServerError):
    """上游 API 错误（通用）"""
    def __init__(self, message: str, code: int = 502, status: str | None = None, details: dict[str, Any] | None = None, upstream_response: str | None = None):
        super().__init__(message, code, status or ErrorStatus.INTERNAL.value, details, upstream_response)

class UnavailableError(ServerError):
    """服务不可用错误 (503)"""
    def __init__(self, message: str = "Service unavailable", details: dict[str, Any] | None = None, upstream_response: str | None = None):
        super().__init__(message, 503, ErrorStatus.UNAVAILABLE, details, upstream_response)

def raise_for_status(
    code: int | str,
    status: str | None = None,
    message: str = "Unknown error",
    details: dict[str, Any] | None = None,
    upstream_response: str | None = None
) -> VertexError:
    """
    根据 HTTP 状态码或 gRPC 状态字符串创建对应的错误实例
    """
    
    try:
        norm_code = int(code)
    except (ValueError, TypeError):
        norm_code = code

    
    
    if status == ErrorStatus.RESOURCE_EXHAUSTED or norm_code == 8 or norm_code == 429:
        return RateLimitError(message, details, upstream_response=upstream_response)
    if status == ErrorStatus.UNAUTHENTICATED or norm_code == 16 or norm_code == 401:
        return AuthenticationError(message, details, upstream_response=upstream_response)
    if status == ErrorStatus.PERMISSION_DENIED or norm_code == 7 or norm_code == 403:
        return PermissionDeniedError(message, details, upstream_response=upstream_response)
    if status == ErrorStatus.INVALID_ARGUMENT or norm_code == 3 or norm_code == 400:
        return InvalidArgumentError(message, details, upstream_response=upstream_response)
    if status == ErrorStatus.NOT_FOUND or norm_code == 5 or norm_code == 404:
        return NotFoundError(message, details, upstream_response=upstream_response)
    if status == ErrorStatus.UNAVAILABLE or norm_code == 14 or norm_code == 503:
        return UnavailableError(message, details, upstream_response=upstream_response)

    
    if isinstance(norm_code, int):
        if 400 <= norm_code < 500:
            return ClientError(message, norm_code, status, details, upstream_response)
        return ServerError(message, norm_code, status, details, upstream_response)
    
    return VertexError(message, status=status, details=details, upstream_response=upstream_response)

def parse_error_response(response_data: str | dict[str, Any] | list[Any]) -> VertexError | None:
    """
    从上游响应中解析错误 (支持 gRPC 风格的 JSON 响应)
    """
    if isinstance(response_data, str):
        try:
            response_data = json.loads(response_data)
        except json.JSONDecodeError:
            return None

    
    if isinstance(response_data, list):
        for item in response_data:
            err = parse_error_response(item)
            if err: return err
        return None

    if not isinstance(response_data, dict):
        return None

    
    if 'error' in response_data:
        err_obj = response_data['error']
        if isinstance(err_obj, dict):
            return raise_for_status(
                code=err_obj.get('code', 500),
                status=err_obj.get('status'),
                message=err_obj.get('message', 'Unknown error'),
                details=err_obj.get('details'),
                upstream_response=json.dumps(response_data)
            )

    
    if 'errors' in response_data:
        errors = response_data['errors']
        if isinstance(errors, list) and len(errors) > 0:
            first_err = errors[0]
            if isinstance(first_err, dict):
                
                ext_status = first_err.get('extensions', {}).get('status', {})
                code = ext_status.get('code') or first_err.get('code', 500)
                status = ext_status.get('status') or first_err.get('status')
                message = ext_status.get('message') or first_err.get('message', 'Unknown error')
                
                return raise_for_status(
                    code=code,
                    status=status,
                    message=message,
                    details=first_err.get('details'),
                    upstream_response=json.dumps(response_data)
                )

    
    if 'code' in response_data or 'status' in response_data or 'message' in response_data:
        return raise_for_status(
            code=response_data.get('code', 500),
            status=response_data.get('status'),
            message=response_data.get('message', 'Unknown error'),
            details=response_data.get('details'),
            upstream_response=json.dumps(response_data)
        )

    return None
