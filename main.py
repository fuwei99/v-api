
import asyncio
import uvicorn

from src.core import (
    load_config,
    PORT_API,
)
from src.api import VertexAIClient, create_app
from src.core.auth import api_key_manager
from src.utils.logger import get_logger, configure_logging, set_request_id


logger = get_logger(__name__)

async def main() -> None:
    """
    启动并运行 Vertex AI Proxy 服务器。
    
    流程：
    1. 加载全局配置。
    2. 初始化 API 密钥管理器（加载有效的客户端密钥）。
    3. 创建 VertexAIClient 客户端用于上游交互。
    4. 构建 FastAPI 应用并挂载路由。
    5. 使用 Uvicorn 启动异步服务器。
    6. 捕获中断信号并安全清理资源。
    """
    
    set_request_id("startup")
    
    config = load_config()
    debug_mode = config.get("debug", False)
    
    logger.info("=" * 60)
    logger.info("🚀 Vertex AI Proxy 启动中...")
    logger.info("📋 模式: Anonymous HTTP")
    logger.info(f"🔧 调试模式: {'开启' if debug_mode else '关闭'}")
    logger.info(f"🌐 API 端口: {PORT_API}")

    
    logger.debug("初始化 API 密钥管理器")
    api_key_manager.load_keys()

    from src.api.admin import ensure_admin_password
    ensure_admin_password()

    from src.transport.codec import needs_worker
    from src.transport.worker import worker

    saved_uri = str(config.get("active_node_uri") or "").strip()
    saved_name = str(config.get("active_node_name") or "")
    if saved_uri and needs_worker(saved_uri):
        try:
            proxy_url = worker.start_with_uri(saved_uri, name=saved_name)
            logger.success(f"✅ 已自动恢复上次的代理节点: {saved_name or saved_uri[:40]} → {proxy_url}")
        except Exception as e:
            logger.warning(f"⚠ 自动恢复代理节点失败: {e}")
    
    logger.debug("创建 Vertex AI 客户端")
    vertex_client = VertexAIClient()
    
    logger.debug("创建 FastAPI 应用")
    app = create_app(vertex_client)
    
    logger.info(f"启动 HTTP API 服务器 (端口: {PORT_API})")
    uvicorn_config = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=PORT_API, 
        log_level="info",
        log_config=None  
    )
    server = uvicorn.Server(uvicorn_config)
    
    logger.success("✅ 服务启动完成，系统运行中...")
    logger.info("=" * 60)
    
    try:
        await server.serve()
    except asyncio.CancelledError:
        logger.info("收到取消信号，开始关闭服务...")
    except KeyboardInterrupt:
        logger.info("收到中断信号 (Ctrl+C)，开始关闭服务...")
    finally:
        logger.info("🛑 开始清理资源...")
        if hasattr(server, 'force_exit'):
            server.force_exit = True
        
        logger.debug("关闭 Vertex AI 客户端")
        await vertex_client.close()

        logger.success("✅ 资源清理完成，服务已安全关闭")

def main_sync() -> None:
    from src.core.config import load_config
    config = load_config()
    configure_logging(debug=config.get("debug", False), log_dir=config.get("log_dir", "logs"))
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main_sync()
