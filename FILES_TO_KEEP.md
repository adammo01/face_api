# 人脸打码维护清单

这个目录里历史部署和测试文件很多。日常只需要关注下面几个文件。

## 核心文件

| 文件 | 用途 |
|---|---|
| `face_blur.py` | 核心检测和打码逻辑, 主要改这里 |
| `app.py` | FastAPI 服务入口, 对外提供 `/api/face_blur` |
| `run_local.py` | 本地批量测试入口 |
| `models/face_detection_yunet_2023mar.onnx` | YuNet 人脸检测模型 |
| `requirements.txt` / `requirements-py39.txt` | 部署依赖 |

## 当前推荐参数

- `mode="gaussian"`: 给 Seedance 当参考图时更自然。
- `score_threshold=0.45`: 比原来的 `0.6` 更偏召回, 小脸和侧脸更容易打到。
- `expand_ratio=0.35`: 比原来的 `0.2` 覆盖更完整, 减少脸颊/下巴/额头边缘漏出。

## 当前检测策略

`face_blur.py` 现在使用 YuNet 多尺度检测:

- 原图检测。
- 额外缩放到长边 `640 / 1024 / 1600` 后再检测。
- 坐标还原后用 NMS 合并。
- Haar 兜底已保留但默认关闭, 因为在拍卖厅/人物全身图上误检较多。

## 可以先忽略的文件

`_deploy_*.py`、`remote_deploy.py`、`_probe_net.py`、`_show_probe.py` 多数是历史部署/排障脚本。没有重新部署服务器时不用看。
