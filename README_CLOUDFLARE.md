# FaceBlur API

Python、OpenCV 和 YuNet 驱动的人脸隐私打码 API。当前仓库保存可复用的图像处理服务与管理页面，Cloudflare 独立部署由 Worker、Container、D1 和 R2 组成。

## 本地运行

```bash
python -m pip install -r requirements.txt
FACE_BLUR_ADMIN_TOKEN=change-me uvicorn app:app --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000/admin`，输入管理员 token 后可查看请求记录、图片库和任务详情。

## 任务接口

- `POST /api/face_blur`：提交处理请求，成功或失败响应均携带 `task_id`。
- `GET /api/tasks/{task_id}`：按任务 ID 查询状态和结果摘要。
- `GET /api/admin/tasks/{task_id}`：管理员查看完整请求、响应、耗时和图片。

完整请求审计会记录 method、path、query、headers 和 body。Authorization、Cookie、token、secret、API key 及 Cloudflare Access JWT 等敏感请求头只保存字段名，值统一为 `[REDACTED]`。

## 测试

```bash
python -m unittest -v test_admin_page.py test_task_tracking.py
```

## 数据目录

- SQLite：`FACE_BLUR_DB_PATH` 或 `FACE_BLUR_DATA_DIR`
- 输出图片：`FACE_BLUR_STATIC_DIR`
- YuNet 模型：`FACE_BLUR_MODEL_DIR`

Cloudflare 版本使用 D1 保存请求记录、R2 保存图片，Container 仅执行 Python/OpenCV 图像处理。
