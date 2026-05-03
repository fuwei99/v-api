

import logging
import sys
import os
import json
import uuid
from datetime import datetime
from typing import Any
from contextvars import ContextVar


request_id_var: ContextVar[str] = ContextVar('request_id', default='')

request_info_var: ContextVar[dict[str, Any]] = ContextVar('request_info', default={})


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

LEVEL_CONFIG = {
    logging.DEBUG: (Colors.DIM + Colors.CYAN, "🔍", "DEBUG"),
    logging.INFO: (Colors.CYAN, "ℹ️ ", "INFO"),
    SUCCESS_LEVEL: (Colors.BRIGHT_GREEN, "✅", "SUCCESS"),
    logging.WARNING: (Colors.BRIGHT_YELLOW, "⚠️ ", "WARN"),
    logging.ERROR: (Colors.BRIGHT_RED, "❌", "ERROR"),
    logging.CRITICAL: (Colors.BOLD + Colors.RED, "💀", "FATAL"),
}


MODULE_ABBR = {
    'vertex_client': 'Vertex',
    'error_logger': 'ErrLog',
    'diff_fixer': 'Diff',
    'processor': 'Stream',
}

class BetterFormatter(logging.Formatter):
    """
    高度定制化的日志格式化器。
    
    功能：
    1. 彩色输出（支持终端检测）。
    2. 自动缩写模块名称以保持对齐。
    3. 支持显示基于 ContextVar 的 Request ID，方便追踪并行请求。
    4. 自动美化输出 JSON 对象（支持 indent）。
    5. 包含图标和精简的时间戳格式。
    """
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        
        now = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        level_tuple = LEVEL_CONFIG.get(record.levelno, (Colors.WHITE, "•", "LOG"))
        
        level_color = level_tuple[0]
        level_icon = level_tuple[1]
        level_name = level_tuple[2]
        
        
        module = record.name.split('.')[-1]
        module = MODULE_ABBR.get(module, module[:8].capitalize())
        
        
        req_id = request_id_var.get()
        req_id_str = f" {Colors.DIM}|{Colors.RESET} {Colors.YELLOW}{req_id[:8]}{Colors.RESET}" if req_id else ""
        
        
        message = record.getMessage()
        
        
        extra_data = getattr(record, 'extra_data', None)
        if extra_data is not None:
            try:
                
                formatted_json = json.dumps(extra_data, indent=2, ensure_ascii=False, default=str)
                indented_json = "\n".join(f"    {line}" for line in formatted_json.splitlines())
                message += f"\n{indented_json}"
            except Exception as e:
                
                message += f" (JSON序列化失败: {e})"
        
        elif isinstance(record.msg, (dict, list)):
            try:
                formatted_json = json.dumps(record.msg, indent=2, ensure_ascii=False, default=str)
                message = f"\n{formatted_json}"
                
                message = "\n".join(f"    {line}" for line in message.splitlines())
            except Exception:
                
                message = str(record.msg)
        

        
        exc_text = ""
        if record.exc_info:
            exc_text = "\n" + self.formatException(record.exc_info)
            
            exc_text = "\n".join(f"    {line}" for line in exc_text.splitlines())

        if self.use_colors:
            return (
                f"{Colors.DIM}{now}{Colors.RESET} "
                f"{level_color}{level_icon} {level_name:<7}{Colors.RESET} "
                f"{Colors.MAGENTA}[{module:^10}]{Colors.RESET}"
                f"{req_id_str} "
                f"{message}{exc_text}"
            )
        else:
            return f"{now} {level_name:<7} [{module:^10}] {req_id[:8] if req_id else ''} {message}{exc_text}"

class LoggerManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self._initialized = True
        self._log_level = logging.INFO
        self._setup_root_logger()

    def _setup_root_logger(self):
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.handlers.clear()

        
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(BetterFormatter())
        console.setLevel(self._log_level)
        root.addHandler(console)

        
        for logger_name in ['httpx', 'httpcore', 'uvicorn', 'fastapi', 'hpack', 'h2', 'uvicorn.error', 'uvicorn.access']:
            l = logging.getLogger(logger_name)
            l.setLevel(logging.WARNING)
            l.propagate = True 

    def configure(self, debug: bool = False, log_file: str | None = None):
        self._log_level = logging.DEBUG if debug else logging.INFO
        root = logging.getLogger()
        for h in root.handlers:
            h.setLevel(self._log_level)
            
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_h = logging.FileHandler(log_file, mode='w', encoding='utf-8', delay=False)
            file_h.setFormatter(BetterFormatter(use_colors=False))
            file_h.setLevel(self._log_level)
            
            file_h.flush()
            root.addHandler(file_h)

class LoggerAdapter(logging.LoggerAdapter):  
    
    def success(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.log(SUCCESS_LEVEL, msg, *args, **kwargs)
        
    def debug_json(self, label: str, data: Any) -> None:
        """专门用于调试 JSON 数据"""
        if getattr(self.logger, "isEnabledFor", lambda x: False)(logging.DEBUG):
            try:
                
                formatted_data = json.loads(json.dumps(data, default=str)) if not isinstance(data, (dict, list)) else data
                self.logger._log(logging.DEBUG, f"{label}:", (), extra={'extra_data': formatted_data})
            except Exception as e:
                self.logger._log(logging.DEBUG, f"{label} (JSON解析失败: {e})", ())

    def debug_large(self, label: str, data: str) -> None:
        """专门用于调试大文本数据"""
        if getattr(self.logger, "isEnabledFor", lambda x: False)(logging.DEBUG):
            self.logger.debug(f"{label}:\n{data}")

def get_logger(name: str) -> LoggerAdapter:
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, {})


manager = LoggerManager()

def configure_logging(debug: bool = False, log_dir: str = "logs"):
    log_path = os.path.join(log_dir, "app.log") if log_dir else None
    manager.configure(debug=debug, log_file=log_path)

def set_request_id(request_id: str | None = None):
    rid = request_id or uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    return rid

def get_request_id() -> str:
    return request_id_var.get()

def clear_context():
    request_id_var.set('')
    request_info_var.set({})
