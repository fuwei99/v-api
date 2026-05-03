# ---- 构建阶段 ----
FROM python:3.11-slim AS builder

WORKDIR /app

# 只复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .

# 安装依赖到独立目录
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- 运行阶段 ----
FROM python:3.11-slim

WORKDIR /app

# 从构建阶段复制已安装的依赖
COPY --from=builder /install /usr/local

# 复制项目源码
COPY . .

# 暴露默认端口
EXPOSE 2156

# 数据卷：允许从宿主机挂载配置和日志
VOLUME ["/app/config", "/app/logs", "/app/errors"]

# 启动
CMD ["python", "main.py"]
