# 渠道5 — 自动人脸打码方案(完整状态)

> 配套方案文档: `serverless_face_blurring_solution.md`
> 本目录为完整可运行的 PoC + 线上服务 + 客户对接文档。
> 最近一次更新: 2026-08-08 (管理后台、实验室、任务审计与图片预览已上线)

## 一、方案要点

| 项 | 选型 | 理由 |
|---|---|---|
| 架构 | FastAPI + uvicorn (**8 workers**, 16 核)、requests Session 连接池、三层缓存(L1 内存 + L2 DB) | 高并发、低延迟、进程重启缓存不丢 |
| 人脸检测 | OpenCV DNN + **YuNet** + 多尺度(640/1600) + **score_threshold=0.45(偏召回)** | 零 API 费用, 232KB, CPU 实时, 大图先缩到 1920px |
| 打码方式 | 5 种 (`pixelate` / `gaussian` / `solid` / `landmark` / `landmark_whole_face`) | 覆盖传统打码 / 风格化 / 真人过滤绕过 |
| 并发 | **上限 128**(代码放开), 当前运行 64, fd 限制 65535 | Admin 后台即时调整, 无需重启 |
| 缓存 | L1 内存 LRU (5min/200条) + L2 DB `blur_cache` (72h TTL) | 重复打码 0ms 直接返回 |
| 部署 | 拓飞云 `36.133.106.162` (Ubuntu 24.04, systemd 守护) | `--limit-max-requests 5000` 防内存泄漏 |

> 说明: 方案文档里同时推荐了 MediaPipe。在 Python 3.13 环境装 MediaPipe 会触发
> `opencv-contrib-python` 替换 `cv2` 的多个文件,沙箱不允许;PoC 阶段改用 OpenCV 官方
> YuNet 模型,效果与 MediaPipe 相当,部署到 Serverless (一般推荐 Python 3.10/3.11) 后
> 也可以一行 import 切回 `mediapipe.tasks.vision.FaceDetector`。

## 二、文件清单

```
渠道5-智能过人脸/
├── README.md                              # 本文件 (完整状态)
├── serverless_face_blurring_solution.md   # 原始方案文档
├── API_INTEGRATION.md                     # 通用对接文档 (含 curl / Python / JS)
├── API_TUOFEI.md                          # 拓飞云线上对接文档 (客户用)
├── 示例图片.jpg                            # 1 人 3 视角示例
├── test_images/                           # 其它测试图
├── outputs/                               # 处理结果
│   ├── mode_compare/                      # 4 模式对比图
│   │   ├── compare.jpg                    # 4 模式拼图
│   │   ├── pixelate.jpg
│   │   ├── gaussian.jpg
│   │   ├── solid.jpg
│   │   └── landmark.jpg
│   └── ...                                # 其它单图打码结果
├── core/
│   ├── face_blur.py                       # 核心处理模块 (5 种打码 + 多尺度)
│   ├── app.py                             # FastAPI 服务 (uvicorn 入口)
│   └── run_local.py                       # 本地批处理入口
├── deploy/
│   ├── Dockerfile                         # 容器化部署
│   ├── docker-compose.yml
│   ├── nginx.conf                         # Nginx 反代模板
│   ├── deploy.sh                          # 裸机一键部署
│   ├── requirements.txt
│   └── requirements-py39.txt
├── remote/
│   ├── remote_deploy.py                   # 跨服务器 SSH 部署
│   ├── _deploy_final.py ~ _deploy_final5.py  # 历史部署脚本
│   └── _deploy_tuofei.py                  # 拓飞云专用部署流程
├── 拓飞云服务器.txt                         # 36.133.106.162 root 凭据
├── 核云服务器.txt / 湖北暗云.txt          # 备选服务器 (跨境/不适合,已弃用)
└── models/
    └── face_detection_yunet_2023mar.onnx  # YuNet 模型
```

## 三、5 种打码模式对照

| Mode | 视觉 | 推荐场景 | 关键参数 |
|---|---|---|---|
| `pixelate` | 马赛克 (块状) | 传统隐私打码,证件照 | `block_size=15` |
| `gaussian` | 高斯模糊 | 通用隐私,自然过渡 | `ksize=31` |
| `solid` | 纯色遮盖 (黑/自定义色) | 极致隐私,完全破坏面部特征 | `color=(0,0,0)` |
| `landmark` | 5 关键点 × 3×3 红点矩阵 | **Seedance 图生视频推荐**,保留角色形象 | `dot_radius=3, spacing=14` |
| `landmark_whole_face` | 整脸网格 + 关键点附近 5×5 密集点 | **多图 reference_image 强绕过人脸过滤** | `face_grid_step=14, dot_radius=3, grid_n=5` |

> 全部 5 种模式都默认走 **多尺度检测** (`detect_multiscale`),复杂场景不会漏大特写。

### 3.1 landmark vs landmark_whole_face

- `landmark`: 只在 5 个核心关键点 (左右眼/鼻尖/左右嘴角) 附近打 3×3 稀疏红点,共 5×9=45 点。
  图保留度高,几乎看不出被打码,但对面部"细节指纹"破坏最弱。
- `landmark_whole_face`: 先在整脸矩形区域按 `face_grid_step=14` 步长铺一层面点,再在关键点附近
  叠加 `grid_n=5` 密集矩阵 (5×5=25 点/关键点,共 5×25=125 点)。整脸区域被破坏,真人过滤更难命中。

**实测选择**:
- 单图 first_frame (视频生成引擎只把这一帧当首帧) → 用 `landmark` 即可
- 多图 reference_image (引擎多帧对比) → 真人检测更严,推荐 `landmark_whole_face`,或 `solid` + `expand_ratio=0.8`

## 四、本地运行

```bash
# 1. 创建 venv (只需一次)
"C:\Users\Administrator\.workbuddy\binaries\python\versions\3.13.12\python.exe" \
    -m venv C:\Users\Administrator\.workbuddy\binaries\python\envs\faceblur

# 2. 装依赖 (清华镜像)
C:\Users\Administrator\.workbuddy\binaries\python\envs\faceblur\Scripts\python.exe \
    -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    opencv-python numpy Pillow

# 3. 处理默认示例图
cd 渠道5-智能过人脸
.../faceblur/Scripts/python.exe run_local.py

# 4. 处理指定目录
.../python.exe run_local.py --dir ../some_img_dir

# 5. 调阈值 / 扩边
.../python.exe run_local.py --score 0.5 --expand 0.3
```

## 五、效果验证

### 5.1 单图场景(示例图片.jpg,1 人 3 视角)

共检测出 **3 张人脸**:

| 位置 | 坐标 (x,y,w,h) | 分数 |
|---|---|---|
| 中间特写 | (412, 175, 115, 155) | 0.94 |
| 左侧坐姿 | (1142, 336, 320, 441) | 0.93 |
| 右侧被头发遮挡的侧面 | (2020, 211, 81, 134) | 0.86 |

### 5.2 复杂场景(测试图,1 人 8 视角 + 一张超大特写)

| 阶段 | 检测数 | 备注 |
|---|---|---|
| 单尺度 | 7 | 大特写脸被漏检(超模型 anchor 上限) |
| 多尺度 | **8** | 缩小图二次检测 + NMS 合并,大特写被识别 |

新增大特写坐标: (324, 276, 449, 602),分数 0.95。

### 5.3 单图耗时(本地 CPU)

| 模式 | 单图(3 脸) | 单图(8 脸 + 多尺度) |
|---|---|---|
| pixelate (马赛克) | ~119 ms | ~241 ms |
| gaussian (高斯模糊) | ~92 ms | ~168 ms |
| solid (纯色遮盖) | ~97 ms | ~160 ms |
| landmark (5 关键点) | ~110 ms | ~210 ms |
| landmark_whole_face | ~140 ms | ~270 ms |

多尺度额外开销约 +50ms,可接受;复杂场景必须开。

### 5.4 4 模式对比图

`outputs/mode_compare/compare.jpg` 一张图看完 4 种风格的差异。

## 六、API 接口

### 6.1 `face_blur.process_image(input_bytes, mode, **params) -> dict`

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `input_bytes` | bytes | 必填 | 任意常见格式图片 |
| `mode` | str | `pixelate` | `pixelate` / `gaussian` / `solid` / `landmark` / `landmark_whole_face` |
| `score_threshold` | float | 0.45 | 人脸分数阈值, 越低召回越多 (隐私打码偏召回) |
| `expand_ratio` | float | 0.2 | 人脸框向外扩比例, 避免边缘漏脸 |
| `**blur_params` | | | `block_size` / `ksize` / `color` / `dot_radius` / `spacing` / `face_grid_step` / `grid_n` |

返回值:

```json
{
  "image_bytes": "...",   // JPEG 二进制
  "format": "jpg",
  "mode": "pixelate",
  "faces": [{"x":..,"y":..,"w":..,"h":..,"score":..}, ...],
  "face_count": 3,
  "elapsed_ms": 118.7
}
```

### 6.2 FastAPI 服务 (`app.py`)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```

| 端点 | 方法 | 说明 |
|---|---|---|
| `POST /api/face_blur` | 接 `{image_url, mode, ...}` | 返回 `{ok, blocked, face_count, elapsed_ms, original_url, output_url, ...}` |
| `GET /healthz` | 健康检查 | 返回 `{ok, model_loaded, pid}` |
| `GET /static/{filename}` | 静态文件 | 返回打码后的 JPEG |
| `GET /admin` | 管理后台页面 | 查看请求、图片库、清理入口 |
| `GET /api/admin/summary` | 管理接口 | 服务统计、存储统计、最近清理记录 |
| `GET /api/admin/requests` | 管理接口 | 最近请求日志 |
| `GET /api/admin/files` | 管理接口 | 静态图片列表 |
| `POST /api/admin/cleanup` | 管理接口 | 手动删除超过 TTL 的静态图片 |

**关键约定**:
- 当 `face_count=0` (无人脸) 时,`output_url == original_url`,**服务端零下载零存储**
- `output_url` 默认由 `FACE_BLUR_PUBLIC_URL` 环境变量生成,设成 `https://api.juziapi.cc.cd` 即可
- 鉴权: `FACE_BLUR_API_TOKEN` (可选, 放在 `Authorization: Bearer <token>` 头)
- 管理后台鉴权: `FACE_BLUR_ADMIN_TOKEN` (放在后台输入框或 `X-Admin-Token` 请求头)
- **缓存**: 相同 URL+参数的请求 72h 内不重复打码, L2 DB `blur_cache` 表命中直接返回 `cached: true`

### 6.4 缓存架构

三层缓存, 按优先级:
```
请求 → L1 内存 LRU (5min/200条) → 命中? 直接返回
     → L2 DB blur_cache (72h TTL) → 命中? 返回 output_url, cached:true
     → L3 实际打码               → 完成后回写 L1+L2
```

| 层 | 存储 | TTL | 命中耗时 | 重启后 |
|---|---|---|---|---|
| L1 | 进程内存 OrderedDict | 5min | 0ms | 丢失 |
| L2 | SQLite `blur_cache` 表 | 72h | <10ms | 保留 |
| L3 | 真实打码 | — | ~2s | — |

- L2 缓存 key = MD5(image_url + 请求参数 JSON)
- 自动过期清理: 查询时惰性 + 定时器每 1h 扫描 `expires_at`
- 磁盘文件丢失也会触发重新打码

### 6.4 管理后台与日志

新增管理后台不改变原有 `/api/face_blur` 响应结构。上线后访问:

```text
https://api.juziapi.cc.cd/admin
```

能力:

- 请求日志: 记录时间、状态、模式、脸数、耗时、输入 URL、输出文件、错误信息、客户端 IP、User-Agent。
- 请求记录默认每页 20 条,可切换 `10 / 20 / 50 / 100 / 200`。每条任务均保留完整请求 JSON 与响应/执行信息,并显示任务 ID。
- 任务详情: 可按任务 ID 或父任务 ID定位,单独查看完整请求、响应、输入图片和生成结果。
- 成功率: 展示总成功率、24h 成功率、重试请求数。
- 图片库: 单独标签页分页展示,默认每页 10 张,可切换 `10 / 20 / 50 / 100 / 200`。点击图片可打开大图预览,按 `Esc` 或点击遮罩关闭。
- 实验室: 管理页内嵌标签,自动复用管理页授权,支持上传图片或输入公网图片 URL并实时预览打码结果,可将参数同步到全局设置。
- 全局设置: 单独标签页,只显示并发、重试、TTL和打码默认参数,不会混入请求记录或图片库。
- 存储统计: 图片数量、总大小、请求总量、24h 请求量、错误量。
- 清理任务: 后台线程按 `FACE_BLUR_IMAGE_TTL_HOURS` 删除过期图片,也可在后台手动触发。
- 内部重试: 下载和处理阶段默认最多重试 2 次,指数退避;外部 API 入参和返回结构不变。
- 数据存储: SQLite 文件,默认 `logs/faceblur.sqlite3`,无需额外数据库。

推荐环境变量:

```bash
FACE_BLUR_DATA_DIR=/root/faceblur/logs
FACE_BLUR_ADMIN_TOKEN=<强随机后台密码>
FACE_BLUR_IMAGE_TTL_HOURS=72
FACE_BLUR_CLEANUP_INTERVAL_SECONDS=3600
FACE_BLUR_MAX_RETRIES=2
FACE_BLUR_RETRY_BACKOFF_SECONDS=0.6
```

安全说明:

- `/admin` 页面可打开,但 `/api/admin/*` 必须 token 才返回数据。
- 如果未设置 `FACE_BLUR_ADMIN_TOKEN`,公网管理接口默认禁用;只有本机 localhost 可访问。
- 不要把 `FACE_BLUR_ADMIN_TOKEN` 写到对接文档或客户请求示例里。

### 6.3 多尺度检测 (`detect_multiscale`)

默认 `process_image` 走的是 `detect_multiscale`,而非单尺度 `detect`。
原因: YuNet 模型的 anchor 尺寸有限,当图中存在超大脸(占图 ≥ 1/4)时单尺度会漏检。

策略:
- 第一遍: 原图直接检测,捕获小脸(占图 < 1/4 的脸)
- 第二遍(仅当 `max(w,h) > 1024`): 缩到最长边 ≤ 1024 再检测一次,捕获大脸
- IoU NMS 合并 (阈值 0.4) 去重

如要强制单尺度(追求速度),可手动调用 `detector.detect(img)`;多尺度耗时 +50ms 左右。

## 七、线上部署(拓飞云)

### 7.1 服务器

| 项 | 值 |
|---|---|
| 实例 | 拓飞云 VPS (Ubuntu 24.04) |
| IP / 端口 | `36.133.106.162:22` |
| 用户 | `root` |
| 凭据 | `拓飞云服务器.txt` (本地 gitignored) |
| 部署路径 | `/root/faceblur/` |
| Python venv | `/root/faceblur/venv/` (Python 3.12.3) |
| 服务 | `faceblur.service` (uvicorn) + `faceblur-portmap.service` (socat) + `cloudflared.service` (隧道) |

### 7.2 三个 systemd 单元

**faceblur.service** — uvicorn 主体
```ini
[Service]
WorkingDirectory=/root/faceblur
LimitNOFILE=65535
Environment=FACE_BLUR_STATIC_DIR=/root/faceblur/static
Environment=FACE_BLUR_MODEL_DIR=/root/faceblur/models
Environment=FACE_BLUR_ADMIN_TOKEN=<强随机>
ExecStart=/root/faceblur/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 8 --limit-max-requests 5000
Restart=always
RestartSec=3
StandardOutput=append:/var/log/faceblur.log
StandardError=append:/var/log/faceblur.log
```

**faceblur-portmap.service** — socat 端口映射
```ini
# 因为 Cloudflare 端的 ingress 写死 localhost:3000,本地监听 8000
# 用 socat 把 3000 → 8000 透明转发
ExecStart=/usr/bin/socat TCP-LISTEN:3000,reuseaddr,fork TCP:127.0.0.1:8000
```

**cloudflared.service** — Cloudflare Tunnel
```ini
ExecStart=/usr/local/bin/cloudflared tunnel --no-autoupdate run
# 凭据文件 /etc/cloudflared/cert.pem, token 已从原九九云迁移过来
```

### 7.3 公网入口

- 域名: `api.juziapi.cc.cd`
- 客户端 → `https://api.juziapi.cc.cd` → Cloudflare 边缘 → 隧道 (QUIC) → `cloudflared` → `localhost:3000` → `socat` → `localhost:8000` (uvicorn)
- 健康检查: `curl https://api.juziapi.cc.cd/healthz` → `{"ok":true,"model_loaded":true,"pid":32485}`

### 7.4 部署踩坑记录

| 坑 | 解决 |
|---|---|
| 拓飞云 8000 端口不能公网直连 (服务商禁止备用 BGP 线路建站) | 全部走 Cloudflare Tunnel,公网只暴露 443 |
| Cloudflare 端 ingress 写死 `localhost:3000`,但 uvicorn 监听 8000 | socat 3000→8000 透明转发 |
| 直接 OpenSSH `ssh root@... < bash` 喂密码失败 | 用 paramiko 脚本 (见 `remote_deploy.py`) |
| SSH 多次密码失败后被 fail2ban 临时 ban | 重试间隔 30-60s,改 paramiko 自动重连 |
| `/opt/faceblur/` 路径每次新 SSH 连接会被清理 | 改用 `/root/faceblur/` (持久化) |
| apt 锁文件残留导致脚本卡住 | `kill -9` + `rm /var/lib/apt/lists/lock` |
| Python 3.12 venv 装不上 | 装 `python3-venv` (非 `python3.12-venv`) |
| apt 部分包 404 | 切阿里云源 |

### 7.5 服务监测

```bash
# 查 50 行最近日志
ssh root@36.133.106.162 "journalctl -u faceblur -n 50 --no-pager"

# 看进程
ssh root@36.133.106.162 "ps -ef | grep -E 'uvicorn|socat|cloudflared' | grep -v grep"

# 健康检查
curl https://api.juziapi.cc.cd/healthz
```

## 八、Serverless 化部署

`face_blur.handler(event, context)` 已经按 FC/SCF/Lambda 通用签名写好,
入参支持 `image_base64` 或 `image_url`,出参为带 `image_base64` 的 JSON。

### 8.1 阿里云函数计算 (FC)

1. 创建函数: 运行时选 `python3.10` 或 `python3.11`,内存 ≥ 512MB,超时 ≥ 30s
2. **关键**: 把 YuNet 模型打成 Layer / 预置到代码包 —— 否则每次冷启动都要从 GitHub 拉,
   不仅慢还可能因网络抖动失败。建议:
   - 把 `face_detection_yunet_2023mar.onnx` 放到代码包根目录或 OSS 启动时下载
   - 或在函数初始化 (`initialize` 钩子) 一次性加载,后续复用
3. 配置 OSS 触发器: Bucket 事件 `oss:ObjectCreated:PutObject` → 函数入参
4. 函数代码入口设为 `face_blur.handler`

### 8.2 腾讯云云函数 (SCF)

类似 FC,但触发器是 COS Bucket,事件格式 `cos:ObjectCreated:*`。
事件结构里 `Records[0].cos.cosObject.url` 就是图片 URL,
handler 里从 `event['Records']` 读 URL 后 `urllib.request.urlopen` 拉取即可。

### 8.3 AWS Lambda

- 运行时 `python3.11`,内存 ≥ 512MB,超时 ≥ 30s
- 用 Lambda Layer 携带 YuNet 模型,或部署到 `/opt/models/` 目录
- S3 PutObject 事件触发,event 里读 `s3.bucket.name` + `s3.object.key` 后 `boto3` 拉图

### 8.4 部署包大小

| 组件 | 大小 |
|---|---|
| opencv-python-headless | ~40 MB |
| numpy | ~15 MB |
| Pillow | ~3 MB |
| YuNet 模型 | ~340 KB |

**强烈建议** 使用 `opencv-python-headless`(去掉 GUI) + `--no-deps` 精简打包,
并把模型放到 Layer / OSS 启动拉取,代码包控制在 50MB 以内(超出要放 OSS / S3)。

### 8.5 冷启动优化

- 进程级 `_DETECTOR` 单例:首次调用加载模型后,同进程后续调用复用
- 模型加载是 IO + CPU 密集,典型冷启动 0.5~2s(不含下载)
- 开启 Provisioned Concurrency / 预置并发 可把延迟降到 ms 级

## 九、扩边 (expand_ratio) 的作用

人脸检测框往往紧贴五官,如果按原框打码,脸颊边缘 / 耳朵 / 头发轮廓会"漏出"。
默认 `expand_ratio=0.2` 表示向外扩 20%,实践中:

- 证件照类 (人脸占大) → 0.1~0.2 即可
- 全身 / 半身照 (人脸占小) → 0.3~0.5 更安全
- 极致隐私 (多图 reference_image) → 0.8 覆盖整头+肩+部分胸
- 自定义需求: 也可传入 `expand_ratio=0` 严格按检测框

## 十、批量并发

Serverless 天然适合批量:

- 每张图独立触发一次函数调用,平台自动横向扩缩容
- 单张图 100ms 级别处理,512MB 内存单实例可支撑 ~10 QPS
- 万级并发只需调高函数并发上限 (FC / Lambda 默认 100,需提工单)

## 十一、渠道6 集成(火山方舟图生视频)

### 11.1 链路

```
用户图片 → 调用渠道5 face_blur API → 打码图 → 调用渠道6 火山方舟 Seedance → 原图 vs 打码图对比
```

### 11.2 关键发现(2026-08-05 实测)

| 输入 | 真人检测结果 | 视频生成 |
|---|---|---|
| 原图 (first_frame, 单图) | `InputImageSensitiveContentDetected.PrivacyInformation` | **400 拒绝** |
| 打码图 (landmark, 单图) | 通过 | ✅ 成功 (251s, 1.22MB) |
| 原图 (reference_image, 多图) | 4 张图全部报 | 400 拒绝 |
| 打码图 (landmark + 多图) | KANE 仍被报 | 400 拒绝 |
| 打码图 (gaussian + 多图) | 3 张被报 | 400 拒绝 |
| 打码图 (solid + expand_ratio=0.8) | 通过 | ✅ 成功 (227s, 2.74MB) |

### 11.3 关键结论

1. **火山方舟多图 reference_image 模式对真人检测比单图 first_frame 严很多**
2. 真人检测不只看脸,而是看整张图的人/体特征 (脸模糊了,身体/服装/姿势还在 → 还是 "真人")
3. **单图 first_frame**: `landmark` 红点模式就够
4. **多图 reference_image**: 必须 `solid` + `expand_ratio=0.8` (覆盖整头+肩+部分胸) 才彻底破坏"人"特征
5. **solid 版仍保留服装/姿势/构图**,模型"脑补"出新面部 — 角色身份可保持,面部一致性下降

### 11.4 推荐接入模板

```python
from face_blur import process_image

# 单图 first_frame 推荐
img_bytes = download(image_url)
res = process_image(img_bytes, mode="landmark",
                    score_threshold=0.6, expand_ratio=0.3)

# 多图 reference_image 推荐
res = process_image(img_bytes, mode="solid",
                    score_threshold=0.6, expand_ratio=0.8)

# 提交到火山方舟
submit_seedance2(ref_image_url=upload(res["image_bytes"]),
                 prompt="...",
                 model="doubao-seedance-2-0-260128")
```

## 十二、线上故障记录

### 12.1 2026-08-06 23:14 D patch landmark 漏检
- **现象**: task `279dea6935...` 的 7.8MB 图片 `face_count=0`
- **根因**: 优化 D 把 landmark 模式的多尺度检测换成了单尺度 `detect_with_landmarks`，resize 1920px 后人脸 score 不达 0.6 漏检
- **修复**: 恢复 `detect_multiscale` 做鲁棒检测 + per-face `detect_with_landmarks` 取关键点
- **同时修复**: `_ck` 作用域 bug — `_face_blur_impl` 不是嵌套函数，访问不到 `face_blur()` 里的 `_ck`，缓存写入实际未生效；改为显式参数 `cache_key` 传递

### 12.2 2026-08-06 18:00 两次 500 错误

观察到的错:
```
OverflowError: cannot convert float infinity to integer
cv2.error: OpenCV(5.0.0) ... in function 'forwardGraph'
              error: (-215:Assertion failed) buf.shape() == m.shape()
```

**根因**:
1. `OverflowError` — `expand_ratio` 太大时坐标裁剪溢到 `np.inf`,进了 OpenCV 整数参数
2. `OpenCV DNN shape mismatch` — YuNet 在极端尺寸下 forward 报错

**当前状态**:
- 服务有自愈,下一次请求 200 OK
- 修复方案: 在 `face_blur.py` 第 491 行附近坐标运算时加 `np.clip(..., 0, img.shape[1])` 兜底

**后续**:
- 长期方案: 异常输入直接返回 400 而不是 500,避免无效重试
- 监控: 加 `prometheus_client` 暴露 5xx 计数,挂在 Cloudflare 端分析

### 12.2 已下线服务器

| 机器 | IP | 原因 |
|---|---|---|
| 核云 (美国洛杉矶) | 156.233.226.5 | 跨境物理不通,SSH 都连不上 |
| 湖北暗云 | 36.133.106.162 | "Ubuntu 24.04" 实际是 Debian 11,环境不一致 |
| 九九云 (Debian 11) | 103.236.84.213 | 能部署,但服务商禁止备用 BGP 线路建站,8000 端口不能公网直连,需 Cloudflare Tunnel |

**最终方案**: 拓飞云 `36.133.106.162` (Ubuntu 24.04) + Cloudflare Tunnel

## 十三、对接文档

- `API_INTEGRATION.md` — 通用对接文档,curl / Python / JS 三种调用方式,完整字段表
- `API_TUOFEI.md` — 拓飞云线上对接文档,精简版,给客户看

## 十四、运维检查清单

```bash
# 1. 服务状态
ssh root@36.133.106.162 "systemctl status faceblur faceblur-portmap cloudflared"

# 2. 健康检查
curl https://api.juziapi.cc.cd/healthz

# 3. 端到端测试
curl -X POST https://api.juziapi.cc.cd/api/face_blur \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/test.jpg", "mode": "landmark"}'

# 4. 查日志
ssh root@36.133.106.162 "journalctl -u faceblur -n 100 --no-pager"

# 5. 端口
ssh root@36.133.106.162 "ss -tlnp | grep -E '3000|8000'"
```

## 十五、版本历史

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-07-24 | v1.0 | 初始 3 模式 (pixelate/gaussian/solid),单尺度 |
| 2026-08-04 | v1.1 | 多尺度检测,修复 8 脸图漏大特写 |
| 2026-08-05 | v1.2 | 新增 `landmark` 点阵模式,适配 Seedance 单图 |
| 2026-08-05 | v1.3 | 新增 `landmark_whole_face` 强模式,适配多图 reference_image |
| 2026-08-06 | v1.5 | **全面优化**: workers 1→8、fd 1024→65535、并发 cap 32→128、requests 连接池、L1+L2 三层缓存(72h DB 持久)、score_threshold 0.6→0.45、大图 resize 1920px、多尺度 4→2、landmark 检测鲁棒性恢复、部署规范固化 |
| 2026-08-08 | v1.6 | 管理后台标签整理、全局设置与实验室修复、任务详情完整审计、请求记录默认 20 条、图片库分页与点击放大预览、实验室 CDN URL 直连下载 |
| 2026-08-05 | v1.4 | 拓飞云部署 + Cloudflare Tunnel + socat 端口映射 |
| 2026-08-05 | v1.4 | 加入 `solid` 极致打码作为多图 fallback |
