# face_blur API · 对接文档

> 单文件服务:仅做人脸打码。输入图片 URL,返回打码后图片的公网 URL;若图中**未检测到人脸**,**直接返回原 URL**(零成本、不下载/不上传)。
> 最新更新:2026-08-07 (新增 parent_task_id 批次标记、图片尺寸展示、Admin 增强)

## 0. 当前线上地址(2026-08-06)

两套部署共用同一份 `渠道5-智能过人脸/app.py`,API 完全一致,域名任选一个即可:

| 部署 | 域名 | 鉴权来源 |
|---|---|---|
| **Cloudflare Workers + Container**（推荐） | `https://api.vpsmo.cc.cd` | 通过 `wrangler secret` 注入（见 §5） |
| SystemD + cloudflared（拓飞云 36.133.106.162） | `https://api.juziapi.cc.cd` | 通过 `FACE_BLUR_ADMIN_TOKEN` 环境变量 |

两个域名都接入了 `/admin` HTML 后台(同一份 ADMIN_HTML)。

**鉴权 token 速查**(完整说明见 §5):

| Token 名 | 用于 | Header | 存放位置 |
|---|---|---|---|
| `API_TOKEN` | `POST /api/face_blur` | `X-Api-Key: <token>` 或 `Authorization: Bearer <token>` | `cloudflare_workers/.secrets.local` + Cloudflare Worker Secret |
| `ADMIN_TOKEN` | `/api/admin/*` 与 `/admin` 页 | `X-Admin-Token: <token>` | 同上 |
| `FACE_BLUR_API_TOKEN` | 本地 Python 版鉴权（可选） | `Authorization: Bearer <token>` | `渠道5-智能过人脸/.env` 或 systemd service 环境 |
| `FACE_BLUR_ADMIN_TOKEN` | 本地 `/admin` 页与 `/api/admin/*` | `X-Admin-Token: <token>` | systemd service 环境 |

## 1. 接口

### `POST /api/face_blur`

**请求体**(JSON)

| 字段 | 必填 | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| `image_url` | ✅ | string(URL) | — | 公网可访问的图片 URL,支持 jpg/png/webp |
| `mode` | ❌ | string | `gaussian` | `pixelate` / `gaussian` / `solid` / `landmark` / **`landmark_whole_face`** |
| `score_threshold` | ❌ | float | `0.52` | 人脸分数阈值(0.1~0.99, 隐私打码偏召回) |
| `expand_ratio` | ❌ | float | `0.30` | 框外扩比例(0~1, 推荐 0.30) |
| `dot_radius` | ❌ | int | `3` | landmark 模式:红点半径 (推荐 3) |
| `spacing` | ❌ | int | `14` | landmark 模式:点阵间距 (推荐 14) |
| `face_grid_step` | ❌ | int | `14` | **`landmark_whole_face` 模式专用**, 整脸网格步长 |
| `grid_n` | ❌ | int | `5` | **`landmark_whole_face` 模式专用**, 关键点附近矩阵大小 |
| `parent_task_id` | ❌ | string | — | **上游任务ID**(max 200字符), 标记同一批图片, 可通过 Admin 面板一键筛选 |

> **所有打码参数都可客户端覆盖传参**(只需在 body 里写明即可),默认值就是 v14r3 推荐参数。

**响应**(JSON)

```json
{
  "ok": true,
  "blocked": true,
  "face_count": 3,
  "elapsed_ms": 124.5,
  "process_ms": 98.2,
  "parent_task_id": "batch-20260807-001",
  "mode": "gaussian",
  "original_url": "https://...",
  "output_url": "https://your-domain/static/xxx.jpg",
  "size": 234567
}
```

**blocked=false(无人脸)** 时的 output_url == original_url,**服务端零下载、零存储**。

### 批量标记（parent_task_id）

同一批图片并发请求时,传入相同的 `parent_task_id` 即可标记为同一批次:
```json
{"image_url": "...", "parent_task_id": "workflow-123", "mode": "landmark_whole_face"}
```

之后在 Admin 面板搜索该 parent_task_id 可一次性查看整批任务结果,便于追踪和审计。**不传则不记录,完全可选。**

### 鉴权(线上 `api.vpsmo.cc.cd` / `api.juziapi.cc.cd` 强制)

每次响应都带 `task_id`,云端 Worker 会把它写进 D1 `requests` 表。
**调用方至少需要 `API_TOKEN`**,不带 header 直接 401:

```bash
curl -X POST https://api.vpsmo.cc.cd/api/face_blur \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: <API_TOKEN>" \
  -d '{"image_url":"https://...","mode":"gaussian"}'

# 也支持 Authorization: Bearer
curl -X POST https://api.vpsmo.cc.cd/api/face_blur \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <API_TOKEN>" \
  -d '{"image_url":"https://...","mode":"gaussian"}'
```

`API_TOKEN` 在 Cloudflare Worker 上通过 `wrangler secret put API_TOKEN` 注入;
本地 Python 版通过 `FACE_BLUR_API_TOKEN` 环境变量注入(留空则不强制)。

### `GET /api/tasks/{task_id}`（公开轮询接口）

异步模式下客户端可用 `task_id` 轮询结果,**不要求 ADMIN_TOKEN**(只要拿到
`task_id` 即可查;task_id 是 32 位 hex,256 位熵,无法枚举)。

```bash
curl https://api.vpsmo.cc.cd/api/tasks/e39b5f3eb6104f53b0ad0dd82e576c52
```

**响应**(JSON)

```json
{
  "task_id": "e39b5f3eb6104f53b0ad0dd82e576c52",
  "created_at": "2026-08-06T01:03:18.660Z",
  "status": "ok",
  "mode": "landmark_whole_face",
  "blocked": 1,
  "face_count": 3,
  "elapsed_ms": 3351.3,
  "process_ms": 2144.4,
  "parent_task_id": "batch-20260807-001",
  "output_url": "https://api.vpsmo.cc.cd/files/outputs/2026/08/06/<task_id>.jpg",
  "error": null,
  "attempts": 1,
  "retried": 0
}
```

字段含义:
- `parent_task_id`: 上游任务批次标记(可能为空)
- `status` = `ok` / `container_error`(服务端进程级异常)
- `blocked` = 1 表示做了打码;0 表示未检测到人脸,服务端零成本直接返回原图
- `output_url` 命中 Cloudflare R2 bucket,通过 `/files/<key>` 公开访问
- `attempts` / `retried` 是容器内重试统计
- 找不到 task_id 返回 `404 {"error": "task not found"}`

### `GET /api/admin/tasks/{task_id}`（管理接口，完整审计）

需要 `ADMIN_TOKEN`,返回该任务的**完整请求/响应/审计**记录(headers、body、
response、client_ip、user_agent):

```bash
curl -H "X-Admin-Token: <ADMIN_TOKEN>" \
  https://api.vpsmo.cc.cd/api/admin/tasks/e39b5f3eb6104f53b0ad0dd82e576c52
```

`request_json` 字段里所有 `Authorization / Cookie / Token / Secret / API-Key / CF-Access-JWT` 头值都被打码为 `[REDACTED]`。

### `GET /api/admin/requests`（管理接口，任务列表 + 批次筛选）

需要 `ADMIN_TOKEN`,分页列出最新任务。可传 `parent_task_id` 参数筛选同一批次:

```bash
# 查看所有任务
curl -H "X-Admin-Token: <ADMIN_TOKEN>" \
  "https://api.vpsmo.cc.cd/api/admin/requests?limit=20&offset=0"

# 按父任务ID筛选同一批次
curl -H "X-Admin-Token: <ADMIN_TOKEN>" \
  "https://api.vpsmo.cc.cd/api/admin/requests?parent_task_id=workflow-123&limit=50"
```

`limit` 默认 10 / 上限 200,`offset` 默认 0。返回 `{items:[...], total, offset, limit}`。

### `GET /healthz`

```json
{"ok": true, "model_loaded": true, "pid": 12345}
```

### `GET /static/{filename}`

返回打码后的 JPEG(公网可直接访问)。

## 2. 五种打码模式

| mode | 说明 | 视觉效果 | 推荐场景 |
|---|---|---|---|
| `pixelate` | 马赛克(默认 block_size=15) | 像素块 | 新闻报道、传统打码 |
| `gaussian` | 高斯模糊(默认 ksize=31) | 柔和虚化 | 通用场景 |
| `solid` | 纯色遮盖(默认黑色) | 完全黑块 | 警情通报、绝对隐私 |
| `landmark` | 关键点点阵(红色,默认半径 3,间距 16) | 只遮眼/鼻/嘴 | 送 Seedance 单图 first_frame (能过审,但多图 reference 不够) |
| **`landmark_whole_face`** | **整脸范围均匀网格打红点 + 关键点附近 grid_n×grid_n 密集叠加**(默认 step=14, r=3, grid_n=5, expand=0.30) | 整脸密集小红点,五官轮廓还透出 | **送 Seedance 多图 reference 模式强烈推荐**(实测能过 `InputImageSensitiveContentDetected`) |

### `landmark_whole_face` 详细参数(2026-08-05 实测过审)

```json
{
  "mode": "landmark_whole_face",
  "dot_radius": 3,
  "spacing": 14,
  "face_grid_step": 14,
  "grid_n": 5,
  "expand_ratio": 0.30,
  "score_threshold": 0.52
}
```

**实测效果**:

| 测试 | 任务 | 状态 | 耗时 | tokens | 大小 |
|---|---|---|---|---|---|
| 480p 4s | `cgt-20260805174311-5fxhx` | ✅ succeeded | 346s | 40,594 | 749 KB |
| 720p 4s | `cgt-20260805175357-9x7g6` | ✅ succeeded | 213s | 87,300 | 2.3 MB |

**为什么能过审**:
- 整脸 ~3500 红点/脸 + 关键点附近 125 红点密集叠加 (~3625 红点/脸)
- 单点小 (r=3),不破坏服装/姿势/构图
- 比 `solid` 整头+肩黑块失真小,模型"脑补"出的面部更接近原角色
- 比 `landmark` 仅关键点小区域覆盖广,真人检测不只看脸

## 2.1 `landmark_whole_face` 打码强度档次(2026-08-05 实测)

> **所有参数都可由客户端覆盖**,下表只是按档列出推荐组合,你可以根据"过审率 vs 面部失真"权衡选哪一档。

| 档次 | dot_radius | spacing | face_grid_step | grid_n | expand_ratio | 单脸红点数 | 面部失真 | 过审率(实测) |
|---|---|---|---|---|---|---|---|---|
| **极轻 v20r3** (基础单图) | 3 | 20 | 20 | 3 | 0.20 | ~1500 + 45 | ⭐⭐ (脸型清晰) | 单图过, 多图可能被拒 |
| **轻 v18r4** (推荐基线) | 4 | 18 | 18 | 4 | 0.25 | ~2300 + 80 | ⭐⭐⭐ | 单图稳过, 多图部分拒 |
| **中 v16r4** (进阶) | 4 | 16 | 16 | 4 | 0.25 | ~2900 + 80 | ⭐⭐⭐⭐ | 多数场景可过 |
| **⭐ 推荐 v14r3** (默认) | **3** | **14** | **14** | **5** | **0.30** | **~3550 + 125** | ⭐⭐⭐⭐⭐ | **多图 reference 实测稳过** |
| **重 v14r4** (强化) | 4 | 14 | 14 | 5 | 0.30 | ~3550 + 125 | ⭐⭐⭐⭐⭐ (密集) | 必过, 但开始影响角色形象 |
| **极重 v12r4** (极致) | 4 | 12 | 12 | 5 | 0.35 | ~4800 + 125 | ⭐⭐⭐⭐⭐⭐ (几乎失真) | 必过, 角色特征破坏大 |

**档次数值含义**:
- `dot_radius`: 单个红点半径 (px),越大点越明显
- `spacing`: 关键点附近叠加点阵的行列间距 (px),越小越密
- `face_grid_step`: 整脸范围均匀网格的步长 (px),越小覆盖越广
- `grid_n`: 关键点附近矩阵大小 (3=3x3, 5=5x5, 7=7x7),越大叠加越密
- `expand_ratio`: 人脸框外扩比例 (0~1),越大覆盖到周边越多

**单图 vs 多图场景差异**:

| 场景 | 推荐档次 | 备注 |
|---|---|---|
| **单图 first_frame** (1 张图 reference) | 轻 v18r4 或 中 v16r4 | 模型"脑补"面部能力强,无需极致打码 |
| **多图 reference** (5 张图同时) | **推荐 v14r3** (默认) 或 更重 | 多图 reference 模式真人检测严 |
| **极致隐私** (绝不能识别) | 重 v14r4 或 极重 v12r4 | 失真大, 但角色身份无法保留 |

**横向对比其它方案(供换档参考)**:

| 方案 | 视觉效果 | 多图 reference 过审 |
|---|---|---|
| `landmark` (关键点 3×3, dot=3, spacing=16) | 只遮 5 关键点小区域 | ❌ KANE 被报 |
| `landmark_whole_face` v14r3 (推荐) | 整脸密集小红点 | ✅ **稳过** |
| `solid` 整头+肩遮盖 (expand_ratio=0.8) | 整头+肩+部分胸黑块 | ✅ 必过, 但面部完全失忆 |

## 3. 调用示例

### 3.1 cURL

```bash
# 通用打码
curl -X POST https://your-domain.com/api/face_blur \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/photo.jpg",
    "mode": "gaussian"
  }'

# 点阵打码 (推荐用于送 Seedance 的图)
curl -X POST https://your-domain.com/api/face_blur \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/photo.jpg",
    "mode": "landmark",
    "dot_radius": 3,
    "spacing": 16
  }'

# 整脸范围小红点 (多图 reference 模式推荐)
curl -X POST https://your-domain.com/api/face_blur \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/photo.jpg",
    "mode": "landmark_whole_face",
    "dot_radius": 3,
    "spacing": 14,
    "face_grid_step": 14,
    "grid_n": 5,
    "expand_ratio": 0.30
  }'

# 自定义强度 (例: 极重档, 极致隐私)
curl -X POST https://your-domain.com/api/face_blur \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/photo.jpg",
    "mode": "landmark_whole_face",
    "dot_radius": 4,
    "spacing": 12,
    "face_grid_step": 12,
    "grid_n": 5,
    "expand_ratio": 0.35
  }'

# 无需打码 (不需要鉴权也能调,服务端自动检测)
# blocked=false 时,output_url == image_url,客户端拿原图即可
```

### 3.1.1 带鉴权 + 异步轮询(线上推荐)

```bash
# 1. 提交任务,拿到 task_id
RESP=$(curl -sS -X POST https://api.vpsmo.cc.cd/api/face_blur \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: <API_TOKEN>" \
  -d '{
    "image_url": "https://example.com/photo.jpg",
    "mode": "landmark_whole_face"
  }')
TASK_ID=$(echo "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin)["task_id"])')
echo "task_id=$TASK_ID"

# 2. 轮询直到 status != pending(目前是同步响应,马上就有结果)
#    但保留这个模式以便将来改成异步
curl -sS "https://api.vpsmo.cc.cd/api/tasks/$TASK_ID" | python3 -m json.tool

# 3. 用 output_url
OUT=$(curl -sS "https://api.vpsmo.cc.cd/api/tasks/$TASK_ID" | python3 -c 'import sys,json; print(json.load(sys.stdin)["output_url"])')
echo "blurred image: $OUT"
```

### 3.2 Python

```python
import httpx, os

API = "https://api.vpsmo.cc.cd/api/face_blur"
API_TOKEN = os.environ["FACE_BLUR_API_TOKEN"]  # 跟 .env 或 CI secret 拿

def blur(image_url: str, mode: str = "gaussian") -> dict:
    r = httpx.post(
        API,
        headers={"X-Api-Key": API_TOKEN},
        json={"image_url": image_url, "mode": mode},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()

def poll_task(task_id: str) -> dict:
    """拿到 task_id 后再轮询(目前用不到,接口保留供异步化)."""
    r = httpx.get(f"https://api.vpsmo.cc.cd/api/tasks/{task_id}", timeout=30)
    r.raise_for_status()
    return r.json()

result = blur("https://example.com/photo.jpg", mode="landmark")
print(f"task_id={result['task_id']} blocked={result['blocked']} faces={result['face_count']}")
print(f"output: {result['output_url']}")
# 直接用 result['output_url'] 即可
```

### 3.3 JavaScript (浏览器/Node)

```js
const API_TOKEN = '...'; // 来自构建时注入,别硬编码进前端 bundle
const r = await fetch('https://api.vpsmo.cc.cd/api/face_blur', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-Api-Key': API_TOKEN,
  },
  body: JSON.stringify({
    image_url: 'https://example.com/photo.jpg',
    mode: 'landmark',
  }),
});
const data = await r.json();
console.log('task_id:', data.task_id);
console.log('output:', data.output_url);  // 直接用作 img.src
```

## 4. 部署

### 4.1 方式 A:Docker(推荐,5 分钟起服务)

```bash
# 在能跑 docker 的机器上
cd /opt
git clone <your-repo> faceblur   # 或 scp 拷过去
cd faceblur/渠道5-智能过人脸/
docker compose up -d
docker compose logs -f faceblur
```

启动成功后,默认监听 `http://<server-ip>:8000`。

### 4.2 方式 B:裸机部署(无 Docker)

```bash
# 把 渠道5-智能过人脸/ 整个目录拷到目标机器 (scp / rsync)
cd 渠道5-智能过人脸
bash deploy.sh    # 自动装 Python 3.11 + 依赖 + systemd service
```

完成后服务由 systemd 守护,重启自动拉起:
```bash
systemctl status faceblur
journalctl -u faceblur -f
```

### 4.3 域名 + HTTPS

1. 准备域名解析到服务器 IP
2. `certbot --nginx -d your-domain.com` 申请 Let's Encrypt 证书
3. 把 `nginx.conf` 拷到 `/etc/nginx/sites-available/faceblur`
4. `ln -s ../sites-available/faceblur /etc/nginx/sites-enabled/`
5. `systemctl restart nginx`
6. 设置 `FACE_BLUR_PUBLIC_URL=https://your-domain.com` 后重启 faceblur 服务

### 4.4 Cloudflare Tunnel(适用于公网 8000 不可达)

如果服务商禁止公网建站(例如九九云),可使用 Cloudflare Tunnel 让服务器主动出站连到 Cloudflare,无需打开入站端口。

```bash
# 1. 在服务器安装 cloudflared
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | gpg --dearmor -o /usr/share/keyrings/cloudflare.gpg
echo "deb [signed-by=/usr/share/keyrings/cloudflare.gpg] https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" > /etc/apt/sources.list.d/cloudflared.list
apt-get update && apt-get install -y cloudflared

# 2. 登录 Cloudflare(浏览器交互)
cloudflared tunnel login

# 3. 建隧道
cloudflared tunnel create faceblur
cloudflared tunnel route dns faceblur api.juziapi.cc.cd

# 4. 写配置
cat > /etc/cloudflared/config.yml <<'EOF'
tunnel: faceblur
credentials-file: /root/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: api.juziapi.cc.cd
    service: http://127.0.0.1:8000
  - service: http_status:404
EOF

# 5. 安装为系统服务
cloudflared service install
systemctl enable --now cloudflared
systemctl status cloudflared
```

启动后 `https://api.juziapi.cc.cd/healthz` 应直接命中本地 uvicorn,不需要再放行公网 8000。

## 5. 配置项(环境变量)

### 5.1 鉴权相关 token（重点）

| 变量 / Secret | 部署 | 用途 | 如何注入 |
|---|---|---|---|
| **`API_TOKEN`**（Cloudflare Secret） | api.vpsmo.cc.cd | `POST /api/face_blur` 鉴权 | `wrangler secret put API_TOKEN`(账号 0a9acbf49aa20ad13c56f7110ec5c138,容器权限要用 Global API key) |
| **`ADMIN_TOKEN`**（Cloudflare Secret） | api.vpsmo.cc.cd | `/api/admin/*` + `/admin` 页 | `wrangler secret put ADMIN_TOKEN` |
| `FACE_BLUR_API_TOKEN` | 本地 `app.py` FastAPI | 同 API_TOKEN,`Authorization: Bearer <token>` | systemd service 的 `Environment=` 行,或 `.env` |
| `FACE_BLUR_ADMIN_TOKEN` | 本地 `app.py` FastAPI | 同 ADMIN_TOKEN,`X-Admin-Token` | 同上 |

**两份 token 互不相关**——`API_TOKEN` 跟业务调用挂钩,`ADMIN_TOKEN` 跟后台管理挂钩,可以分别给不同的同事/服务。两者都留空就完全开放(只建议本地开发用)。

线上 `api.juziapi.cc.cd`(SystemD 部署)的两个 token 在
`/etc/systemd/system/faceblur.service` 的 `Environment=` 行里:
```ini
Environment=FACE_BLUR_API_TOKEN=...
Environment=FACE_BLUR_ADMIN_TOKEN=...
```

### 5.2 其他环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `FACE_BLUR_STATIC_DIR` | `/app/static` | 打码图存储目录 |
| `FACE_BLUR_MODEL_DIR` | `/models` | YuNet 模型目录(空就下载) |
| `FACE_BLUR_PUBLIC_URL` | (空) | output_url 的前缀,反代后填 `https://your-domain.com` |
| `FACE_BLUR_DL_TIMEOUT` | 30 | 下载超时(秒) |
| `FACE_BLUR_MAX_BYTES` | 20MB | 输入图最大字节数 |

## 6. 性能 & 限制

- 单张图(普通 2000x1500,3 脸):**100~250ms**(纯检测+打码), 端到端 ~700-3000ms (含下载)
- 10 脸密集图:**~750ms** 端到端
- 内存峰值:每 worker 约 ~150MB, 8 workers 合计 ~1.2GB (16 核 / 31GB 服务器)
- 并发上限: 64(可在 Admin 面板调整, 上限 128)
- 去重缓存: 内存 LRU(5 分钟/200 条) + DB 持久缓存(72 小时), 重复请求 0ms 返回
- 输入图最大 20MB(可调)
- 支持自动过期清理、失败重试、SSRF 防护

## 7. 客户端集成模式

### 模式 1:同步直接用

```
用户上传 → 你的服务器 → 调 faceblur API → 拿 output_url → 存到你自己的 OSS
```
最简单,适合小流量。

### 模式 2:接力送给 Seedance

```python
# 1. 打码(landmark 模式)
blurred = blur_api(image_url=user_image, mode="landmark")
# 2. 送 Seedance
seedance_resp = seedance_client.content_generation.tasks.create(
    model="doubao-seedance-2-0-260128",
    content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": blurred["output_url"]},
         "role": "reference_image"},
    ],
    ratio="16:9", duration=5, resolution="480p",
)
```
真人素材图打码后送 Seedance,**绕过 `InputImageSensitiveContentDetected`**,同时保留角色形象,生成的视频角色一致性好。

### 模式 3:无人脸直传

如果 `blocked == false`,说明 API 端没检测到人脸,output_url == original_url,**客户端可直接用 original_url**,不需要二次处理。

## 8. 文件清单(部署包)

```
渠道5-智能过人脸/
├── face_blur.py         # 核心打码模块 (含多尺度检测 + landmark 模式)
├── app.py               # FastAPI 服务
├── requirements.txt     # Python 依赖
├── Dockerfile           # Docker 镜像构建
├── docker-compose.yml   # 一键启动
├── deploy.sh            # 裸机一键部署 (Ubuntu/Debian)
├── nginx.conf           # 反代 + HTTPS 配置模板
├── README.md            # 本文件 (PoC + 方案说明)
├── API_INTEGRATION.md   # 客户端对接文档 (本文档)
├── static/              # 打码图持久化目录 (运行时生成)
└── models/              # YuNet 模型 (启动时下载)
```

## 9. 常见问题

**Q: 输出图能永久保留吗?**
A: 默认在 `static/` 目录,容器重启会丢。建议:
- 生产挂载 volume(已配)
- 或定期同步到 OSS(写个 cron)

**Q: 不打码的图会被服务端存吗?**
A: 不会。无人脸时 `output_url == original_url`,服务端零下载,直接 echo。

**Q: 服务端支持人脸上传吗?**
A: 服务端**只接受 URL**(不接受 multipart),避免你把用户原图传到第三方。客户端先自己上传到自己的 OSS,再传 URL 给我们。

**Q: 怎么计费?**
A: 这个服务**纯本地**,无第三方 API 调用,只算 CPU 时间 + 带宽 + 存储成本。

**Q: 怎么防滥用?**
A: 加 `FACE_BLUR_API_TOKEN`,在 app.py 里启用鉴权中间件。

## 10. 联系与反馈

模型 + 服务都在 `渠道5-智能过人脸/`,所有源码可改可审计。Serverless 部署见 `serverless_face_blurring_solution.md`。