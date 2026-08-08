# face_blur API · 拓飞云版对接文档

> 当前正式环境:拓飞云 `36.133.106.162`(Ubuntu 24.04.4 LTS),通过 Cloudflare Tunnel 暴露为 `https://api.juziapi.cc.cd`。本文档面向**直接对接方**,只讲怎么调,不涉及部署。
> 管理后台: `https://api.juziapi.cc.cd/admin`（仅管理员使用,不属于客户 API）

## 1. 接口地址

| 项 | 值 |
|---|---|
| 域名 | `https://api.juziapi.cc.cd` |
| 协议 | HTTPS(Cloudflare 终结 TLS)+ HTTP/1.1 |
| 健康检查 | `GET /healthz` |
| 人脸打码 | `POST /api/face_blur` |
| 静态文件 | `GET /static/{filename}.jpg` |

无需鉴权、无需 API Key。

## 2. 健康检查

```bash
curl https://api.juziapi.cc.cd/healthz
```

返回:

```json
{"ok": true, "model_loaded": true, "pid": 14833}
```

`model_loaded: true` 表示模型已加载,可正常服务。

## 3. 人脸打码接口

### 3.1 请求

```bash
curl -X POST https://api.juziapi.cc.cd/api/face_blur \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/photo.jpg",
    "mode": "landmark",
    "dot_radius": 3,
    "spacing": 16
  }'
```

### 3.2 请求字段

| 字段 | 必填 | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| `image_url` | ✅ | string(URL) | — | 公网可访问的图片 URL,支持 jpg/png/webp |
| `mode` | ❌ | string | `gaussian` | `pixelate` / `gaussian` / `solid` / `landmark` |
| `score_threshold` | ❌ | float | `0.6` | 人脸分数阈值(0.1~0.99,越低召回越多) |
| `expand_ratio` | ❌ | float | `0.2` | 脸框外扩比例(0~1,默认扩 20%) |
| `dot_radius` | ❌ | int | `3` | landmark 模式:红点半径(像素) |
| `spacing` | ❌ | int | `16` | landmark 模式:点阵间距(像素) |
| `callback_url` | ❌ | string(URL) | — | 异步回调(暂未启用) |

### 3.3 响应(检测到人脸时)

```json
{
  "ok": true,
  "blocked": true,
  "face_count": 1,
  "elapsed_ms": 2378.58,
  "process_ms": 340.97,
  "mode": "landmark",
  "original_url": "https://example.com/photo.jpg",
  "output_url": "https://api.juziapi.cc.cd/static/1785917641_8dc9a104.jpg",
  "size": 416865
}
```

### 3.4 响应(无人脸时)

```json
{
  "ok": true,
  "blocked": false,
  "face_count": 0,
  "elapsed_ms": 12.5,
  "mode": "gaussian",
  "original_url": "https://example.com/no-face.jpg",
  "output_url": "https://example.com/no-face.jpg",
  "message": "no face detected, return original url"
}
```

**关键**:无人脸时 `output_url == original_url`,**服务端零下载零存储**,直接 echo 原 URL。

### 3.5 错误响应

```json
{"detail": "download failed: HTTP Error 404: Not Found"}   // 400
{"detail": "image too large (> 20971520 bytes)"}          // 413
```

- `400`:图片下载失败、URL 格式错误、参数非法
- `413`:图片超过 20MB
- `500`:服务端处理失败

## 4. 四种打码模式

| mode | 视觉效果 | 推荐场景 |
|---|---|---|
| `pixelate` | 像素马赛克(整脸) | 新闻报道、传统打码风格 |
| `gaussian` | 高斯虚化(整脸) | 通用、想保留美观 |
| `solid` | 纯色矩形遮盖(整脸,默认黑) | 警情通报、绝对隐私 |
| `landmark` | **稀疏红色点阵**(只遮眼/鼻/嘴) | **送 Seedance 图生视频时强烈推荐** |

**实际产品建议**:要做下游 AI(Seedance 图生视频等) → `landmark`,做隐私保护 → `solid` 或 `gaussian`。

## 5. 客户端调用示例

### 5.1 cURL

```bash
# landmark(推荐,送下游模型)
curl -X POST https://api.juziapi.cc.cd/api/face_blur \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/photo.jpg","mode":"landmark"}'

# 高斯模糊
curl -X POST https://api.juziapi.cc.cd/api/face_blur \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/photo.jpg","mode":"gaussian"}'

# 黑色遮盖
curl -X POST https://api.juziapi.cc.cd/api/face_blur \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/photo.jpg","mode":"solid"}'
```

### 5.2 Python

```python
import httpx

API = "https://api.juziapi.cc.cd/api/face_blur"

def blur(image_url: str, mode: str = "landmark", **extra) -> dict:
    payload = {"image_url": image_url, "mode": mode, **extra}
    r = httpx.post(API, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

# 调一次
result = blur("https://example.com/photo.jpg", mode="landmark")
print(f"blocked={result['blocked']} faces={result['face_count']}")
print(f"output_url={result['output_url']}")

# blocked=false 时 output_url == image_url,直接当原图用就行
img_url = result["output_url"]  # 客户端永远用这个字段,不判断 blocked
```

### 5.3 JavaScript

```js
async function blur(imageUrl, mode = 'landmark') {
  const r = await fetch('https://api.juziapi.cc.cd/api/face_blur', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_url: imageUrl, mode }),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await r.json();
  return data.output_url;  // 直接用,无需判断 blocked
}

const url = await blur('https://example.com/photo.jpg');
document.querySelector('img').src = url;
```

## 6. 与下游 Seedance 接力

```python
import httpx

# 1. 打码(landmark)
blur_resp = httpx.post(
    "https://api.juziapi.cc.cd/api/face_blur",
    json={"image_url": user_image_url, "mode": "landmark"},
    timeout=60,
).json()

# 2. 把打码图作为首帧送给 Seedance
seedance_resp = httpx.post(
    "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks",
    headers={"Authorization": f"Bearer {ARK_API_KEY}"},
    json={
        "model": "doubao-seedance-2-0-260128",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": blur_resp["output_url"]},
             "role": "reference_image"},
        ],
        "ratio": "16:9", "duration": 5, "resolution": "480p",
    },
    timeout=60,
).json()

print(seedance_resp["id"])
```

## 7. 性能与限制

| 项 | 值 |
|---|---|
| 单张图打码耗时 | 100~340ms(纯 CPU,不含下载原图) |
| 含下载总耗时 | 0.5~3.5s(取决于原图大小和网络) |
| 单实例吞吐 | ~10 QPS |
| 输入图最大 | 20MB |
| 并发限制 | 暂无限流(Cloudflare Tunnel 默认支持) |

## 8. 调用须知

1. **图片必须公网可访问**:API 服务需要主动下载,不能传 base64。
2. **HTTPS 推荐**:虽然 HTTP 也支持,但 HTTPS 能走 Cloudflare 边缘缓存,延迟更低。
3. **output_url 永久有效**:URL 对应的文件保存在 `static/`,容器重启不丢(理论上),但建议客户端**尽快消费并保存到自己的 OSS**。
4. **不要缓存 input URL 的检测结果**:同一张图反复调用会重复下载 + 打码,服务端不做去重。

## 9. 环境信息(运维参考)

| 项 | 值 |
|---|---|
| 服务器 | 拓飞云 36.133.106.162(广州移动) |
| 系统 | Ubuntu 24.04.4 LTS |
| Python | 3.12.3 |
| uvicorn 端口 | 0.0.0.0:8000(单 worker) |
| 公网映射 | Cloudflare Tunnel(`api.juziapi.cc.cd`) |
| 入口来源 | socat 127.0.0.1:3000 → 127.0.0.1:8000 |
| 模型 | `face_detection_yunet_2023mar.onnx`(~227KB) |
| 检测器 | OpenCV DNN + YuNet 多尺度 |

## 10. 常见问题

**Q:返回的 `output_url` 是相对路径还是完整 URL?**
A: 当前是**完整公网 URL**,形如 `https://api.juziapi.cc.cd/static/xxx.jpg`,可直接使用。

**Q:同一张图能重复调吗?**
A:能,服务端无去重,每次都会重新打码并生成新文件,output_url 每次都不同。

**Q:无打码直接原图会被服务端存吗?**
A:不会。`blocked=false` 时服务端**不下不存**,直接 echo 输入 URL。

**Q:打码图能永久保留吗?**
A:理论上保留(`static/` 是容器内目录),但建议客户端拿到 URL 后立即同步到自己的对象存储。

**Q:能传本地文件吗?**
A:不能,API 只接受公网 URL。需要客户端先上传到自己 OSS,再把 URL 给我们。

**Q:API 出错了找谁?**
A:服务端是纯本地代码 + Cloudflare Tunnel,99% 的错误是图片下载失败(原 URL 不通),不是 API 本身。
