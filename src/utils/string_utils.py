import re

def snake_to_camel(snake_str: str) -> str:
    """
    将下划线命名法（snake_case）转换为小驼峰命名法（camelCase）。
    常用于将 Python 风格的参数映射到 Google API 期望的格式。
    """
    if "_" not in snake_str:
        return snake_str
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def camel_to_snake(camel_str: str) -> str:
    """将 camelCase 字符串转换为 snake_case"""
    
    snake_str = re.sub('([a-z0-9])([A-Z])', r'\1_\2', camel_str).lower()
    return snake_str
