FROM docker.m.daocloud.io/library/python:3.11-slim

# opencv-python-headless 节省 ~40MB, 且不需要 GUI 依赖
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 清华镜像装依赖
COPY requirements.txt /app/requirements.txt
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt

COPY face_blur.py app.py /app/

# YuNet 模型 (启动时下载到 /tmp, 缓存到 volume 提升后续启动速度)
ENV FACE_BLUR_MODEL_DIR=/models
RUN mkdir -p /models && \
    curl -sSL -o /models/face_detection_yunet_2023mar.onnx \
      https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet_2023mar.onnx

# 打码图静态目录
ENV FACE_BLUR_STATIC_DIR=/app/static
RUN mkdir -p /app/static

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
