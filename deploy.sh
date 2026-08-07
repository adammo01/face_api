#!/usr/bin/env bash
# 一键部署脚本 (Ubuntu/Debian)
# 用法: bash deploy.sh
set -e

echo "==== face_blur API 一键部署 ===="

# 1. 装基础依赖
if ! command -v python3.11 >/dev/null 2>&1; then
    echo "[1/4] 装 Python 3.11"
    apt-get update -qq
    apt-get install -y --no-install-recommends software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa || true
    apt-get update -qq
    apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev \
        libgl1 libglib2.0-0 curl ca-certificates
fi

# 2. 建 venv
echo "[2/4] 建 venv"
APP_DIR=/opt/faceblur
mkdir -p "$APP_DIR/static" "$APP_DIR/models"
python3.11 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
"$APP_DIR/venv/bin/pip" install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 复制代码
echo "[3/4] 复制代码"
cp face_blur.py app.py "$APP_DIR/"
[ -f /models/face_detection_yunet_2023mar.onnx ] || \
    curl -sSL -o "$APP_DIR/models/face_detection_yunet_2023mar.onnx" \
      https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet_2023mar.onnx

# 4. systemd service
echo "[4/4] 注册 systemd 服务"
cat > /etc/systemd/system/faceblur.service <<'EOF'
[Unit]
Description=face_blur API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/faceblur
Environment="FACE_BLUR_STATIC_DIR=/opt/faceblur/static"
Environment="FACE_BLUR_MODEL_DIR=/opt/faceblur/models"
Environment="FACE_BLUR_PUBLIC_URL="  # 反代时填 https://your-domain.com
ExecStart=/opt/faceblur/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
Restart=on-failure
RestartSec=3
StandardOutput=append:/var/log/faceblur.log
StandardError=append:/var/log/faceblur.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable faceblur
systemctl restart faceblur
sleep 2
systemctl status faceblur --no-pager

echo
echo "==== 部署完成 ===="
echo "服务地址: http://$(curl -s ip.sb):8000"
echo "健康检查: curl http://localhost:8000/healthz"
echo "日志:     tail -f /var/log/faceblur.log"
echo "重启:     systemctl restart faceblur"