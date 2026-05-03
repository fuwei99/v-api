

import asyncio
import os
import base64
import random
import re
import time
from bs4 import BeautifulSoup
from typing import Any, AsyncGenerator, Optional
from curl_cffi import requests
from curl_cffi.requests import Response
from src.core.config import load_config
from src.core.proxy import get_primary_proxy
from src.utils.logger import get_logger

logger = get_logger(__name__)

def _random_string(length: int) -> str:
    return "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(length))

class NetworkClient:
    """
    底层网络客户端，负责 HTTP 通信及 Google 伪装。
    
    核心功能：
    1. 模拟浏览器指纹（impersonate）以降低被 Google 拦截的概率。
    2. 实现自动化 Recaptcha Enterprise Token 获取流程。
    3. 管理 HTTP 会话与代理配置。
    4. 支持流式和非流式请求。
    """
    
    def __init__(self):
        self.config = load_config()
        self.recaptcha_base_api = "https://www.google.com"
        self.browser_targets = ["chrome124", "chrome131", "chrome146"]
        logger.debug("NetworkClient 初始化完成")

    async def close(self):
        pass 

    def _get_imp(self) -> str:
        return random.choice(self.browser_targets)

    async def fetch_recaptcha_token(self, session: requests.AsyncSession) -> Optional[str]:
        """获取 Google Recaptcha Token"""
        import os
        import base64
        
        for retry in range(3):
            try:
                
                js_url = f"{self.recaptcha_base_api}/recaptcha/enterprise.js?render=6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj"
                js_resp = await session.get(js_url, timeout=15)
                v_match = re.search(r'releases/([\w-]+)/recaptcha', js_resp.text)
                v_version = v_match.group(1) if v_match else "gTpTIWhbKpxADzTzkcabhXN4"

                
                random_cb = _random_string(10)
                anchor_url = (
                    f"{self.recaptcha_base_api}/recaptcha/enterprise/anchor?"
                    f"ar=1&k=6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj&"
                    f"co=aHR0cHM6Ly9jb25zb2xlLmNsb3VkLmdvb2dsZS5jb206NDQz&"
                    f"hl=zh-CN&v={v_version}&size=invisible&badge=inline&"
                    f"anchor-ms=20000&execute-ms=30000&cb={random_cb}"
                )
                
                
                session.cookies.clear()
                
                anchor_response = await session.get(anchor_url, timeout=15)
                soup = BeautifulSoup(anchor_response.text, "html.parser")
                token_element = soup.find("input", {"id": "recaptcha-token"})
                
                if token_element is None:
                    logger.warning(f"anchor_html 未找到 token 元素 (尝试 {retry+1}/3)")
                    continue
                    
                base_recaptcha_token = str(token_element.get("value"))

                
                reload_url = f"{self.recaptcha_base_api}/recaptcha/enterprise/reload?k=6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj"
                
                fake_vh = random_cb 
                
                payload = {
                    "v": v_version, 
                    "reason": "q", 
                    "k": "6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj",
                    "c": base_recaptcha_token, 
                    "co": "aHR0cHM6Ly9jb25zb2xlLmNsb3VkLmdvb2dsZS5jb206NDQz",
                    "hl": "zh-CN", 
                    "size": "invisible",
                    "vh": fake_vh,
                    "chr":"", 
                    "bg": "",  
                }
                
                reload_headers = {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Referer": anchor_url,
                    "Origin": "https://www.google.com"
                }
                
                reload_response = await session.post(
                    reload_url, data=payload, headers=reload_headers, timeout=15
                )
                
                match = re.search(r'rresp","(.*?)"', reload_response.text)
                if not match:
                    logger.warning(f"未找到 rresp (尝试 {retry+1}/3)")
                    continue
                    
                final_token = match.group(1)
                logger.debug("成功获取 Recaptcha Token")
                return final_token
                
            except Exception as e:
                logger.error(f"获取 recaptcha_token 异常 (尝试 {retry+1}/3): {e}")
                    
        logger.error("获取 Recaptcha Token 失败")
        return None

    def create_session(self) -> requests.AsyncSession:
        """创建一个带有随机伪装指纹的 Session"""
        imp = self._get_imp()
        logger.debug(f"创建新 Session (指纹: {imp})")
        proxy_url = get_primary_proxy(self.config)
        
        return requests.AsyncSession(
            impersonate=imp,
            proxies={"http": proxy_url, "https": proxy_url} if proxy_url else None
        )

    async def post_request(self, session: requests.AsyncSession, url: str, headers: dict[str, str], json_data: dict[str, Any]) -> Response:
        """发送非流式 POST 请求"""
        try:
            return await session.post(url=url, headers=headers, json=json_data, timeout=30.0)
        except Exception as e:
            logger.error(f"非流式网络请求异常: {e}")
            raise

    async def stream_request(self, session: requests.AsyncSession, method: str, url: str, headers: dict[str, str], json_data: dict[str, Any]) -> AsyncGenerator[Response, None]:
        """发送流式请求"""
        try:
            response = await session.request(
                method=method, url=url, headers=headers, json=json_data, timeout=180.0, stream=True
            )
            yield response
        except Exception as e:
            logger.error(f"网络请求异常: {e}")
            raise