# 拓飞云 face_blur 服务 — 代码巡检 Bug 清单与修复建议

> 巡检时间：2026-08-07 08:44
> 巡检范围：`渠道5-智能过人脸/app.py` + `face_blur.py`（线上运行版本）
> 前置：上一轮已修复 4 项（_last_size 污染 / landmarks 丢失 / L2 缓存缺字段 / 下载连接未关闭），本轮为新发现的遗留问题

---

## 优先级总览

| 优先级 | 数量 | 说明 |
|--------|------|------|
| 🔴 P0 | 1 | 会导致错误行为，必须修 |
| 🟠 P1 | 4 | 并发/安全风险，建议尽快修 |
| 🟡 P2 | 5 | 稳定性/性能问题，择机修 |
| 🟢 P3 | 7 | 代码质量，低优先 |

---

## 🔴 P0 — 会导致错误行为

### Bug 1：L1 缓存命中直接修改共享 dict，task_id 竞态

**位置**：`app.py` L730-734（face_blur 入口函数）

```python
_cached = _cache_get(_ck)
if _cached is not None:
    _cached["task_id"] = task_id   # ← 直接改缓存里的共享对象
    return _cached
```

**问题**：
- `_RESPONSE_CACHE` 里存的是同一个 dict 对象的引用
- 两个请求同时命中同一缓存时，都往这个 dict 写自己的 `task_id`
- FastAPI 在 `return` 之后才序列化响应，此时 dict 里的 `task_id` 可能已是**另一个请求的值**
- 客户端拿到的 task_id 不是自己这单的 → 用它查 `/api/tasks/{id}` 会 404 或查到别人的记录

**触发条件**：同一图片 URL + 参数 5 分钟内被并发请求（当前 QPS 低，概率不高，但一旦发生难排查）

**修复建议**：
```python
if _cached is not None:
    return {**_cached, "task_id": task_id}   # 拷贝一份再改，不污染缓存
```

---

## 🟠 P1 — 并发 / 安全风险

### Bug 2：`requests.Session` 非线程安全

**位置**：`app.py` `_dl_session()` / `_download()`

**问题**：
- requests 官方文档明确 **Session 不是线程安全的**
- 8 个 uvicorn worker 是进程隔离（每进程一份 Session，没问题）
- 但**单进程内** FastAPI 的多个线程共用同一个 Session 并发 `get`，可能出现连接复用错乱/异常
- 当前 OpenCV 打码是 CPU 密集（持锁），下载是 IO（放锁），实际并发下载可能性存在

**修复建议**（二选一）：
```python
# 方案 A：线程本地 Session（改动最小）
_DL_LOCAL = threading.local()

def _dl_session():
    sess = getattr(_DL_LOCAL, "session", None)
    if sess is None:
        import requests as _rq
        sess = _rq.Session()
        _adapter = _rq.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=10)
        sess.mount("https://", _adapter)
        sess.mount("http://", _adapter)
        _DL_LOCAL.session = sess
    return sess

# 方案 B：换 httpx.Client（线程安全，但要多一个依赖）
```

### Bug 3：L1 缓存 OrderedDict 无锁

**位置**：`app.py` `_cache_get()` / `_cache_set()`

**问题**：
- `_RESPONSE_CACHE` 是模块级 `OrderedDict`
- `move_to_end` + `popitem` + `del` 组合操作不是原子的
- 并发读写可能 `RuntimeError: dictionary changed size during iteration`
- 最坏结果：一次 500 + 缓存失效重新打码（影响小但存在）

**修复建议**：
```python
_CACHE_LOCK = threading.Lock()

def _cache_get(key):
    with _CACHE_LOCK:
        ...  # 原逻辑包进锁

def _cache_set(key, resp):
    with _CACHE_LOCK:
        ...  # 原逻辑包进锁
```

### Bug 4：SSRF 漏洞（安全）

**位置**：`app.py` `_download()` — `image_url` 用户可控

**问题**：
- 服务器会去请求任意用户提供的 URL，可被当作跳板：
  - `http://169.254.169.254/` → 云厂商元数据（可能泄露凭据）
  - `http://127.0.0.1:8000/api/admin/...` → 本机管理接口
  - 内网 IP 扫描
- 当前 API 有 token 鉴权，但 SSRF 风险独立存在（被攻击后 token 也可能被读走）

**修复建议**：下载前校验目标 IP：
```python
import ipaddress, socket

_BLOCKED_NETS = [
    ipaddress.ip_network(n) for n in (
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12",
        "192.168.0.0/16", "169.254.0.0/16", "0.0.0.0/8",
        "::1/128", "fc00::/7", "fe80::/10",
    )
]

def _assert_public_url(url: str):
    host = urllib.parse.urlparse(url).hostname
    for info in socket.getaddrinfo(host, None):
        ip = ipaddress.ip_address(info[4][0])
        if any(ip in net for net in _BLOCKED_NETS):
            raise HTTPException(400, f"blocked private/link-local address: {ip}")
```

### Bug 5：并发上限是"每进程"而非全局

**位置**：`app.py` `_task_slot()` — `_INFLIGHT_TASKS` 进程内计数

**问题**：
- 8 个 worker 进程各自独立计数
- Admin 面板设 `max_concurrent_tasks=64`，实际全局并发是 **8 × 64 = 512**
- 业务方以为 64 是硬上限，实际流量会远超预期
- 大量并发时 SQLite 写锁冲突（见 Bug 8）会放大

**修复建议**（任选）：
1. 文档/Admin 面板明确标注"每进程限额，全局 = 8 × 该值"
2. 用共享计数（Redis / SQLite 表 + 事务）做真正全局限流

---

## 🟡 P2 — 稳定性 / 性能

### Bug 6：`_ensure_model` 并发下载无锁

**位置**：`face_blur.py` `_ensure_model()`

**问题**：多线程首次同时触发模型下载，多个 `urlretrieve` 写同一路径，文件可能损坏，OpenCV 加载报 "Failed to parse ONNX model"（历史上遇到过 v1 镜像的同类问题）

**修复建议**：
```python
_MODEL_LOCK = threading.Lock()

def _ensure_model(...):
    with _MODEL_LOCK:
        if model_path.exists() and model_path.stat().st_size > 1000:
            return model_path
        # 下载逻辑
```

### Bug 7：admin_summary 全量文件扫描 + 双重 stat

**位置**：`app.py` `_static_files()` / `admin_summary()`

**问题**：
- `sorted(glob, key=p.stat())` 全量排序
- 列表构建时又 `path.stat()` 一次 → 每文件 stat 两次
- 2200+ 文件 + 页面 30s 轮询，CPU 浪费

**修复建议**：
```python
# 一次 stat 存入元组，一次遍历
paths = [(p, p.stat()) for p in STATIC_DIR.glob("*.jpg")]
paths.sort(key=lambda x: x[1].st_mtime, reverse=True)
```
长期：给 requests 表加 output_file 索引，用 DB 分页代替目录扫描。

### Bug 8：高并发下 `_insert_request` 丢记录

**位置**：`app.py` `_insert_request()`

**问题**：SQLite WAL 单写者。8 进程 × 64 并发同时 insert 时，busy timeout 10s 内可能 `database is locked`，异常被 `except` 吞掉 → 请求日志丢失（打码结果本身不受影响，仅统计/审计缺失）

**修复建议**：
```python
def _insert_request(row):
    for attempt in range(3):   # 简单重试
        try:
            with _db() as conn:
                conn.execute(...)
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < 2:
                time.sleep(0.05 * (attempt + 1))
                continue
            log.warning(...)
            return
```

### Bug 9：`_get_detector` 按阈值频繁重建 detector

**位置**：`face_blur.py` `_get_detector()`

**问题**：客户端每次传不同 `score_threshold`（0.45 / 0.6 / 0.5...）就重建 FaceDetector（重新加载模型）。混用阈值的请求会让线程的 detector 反复重建，性能抖动。

**修复建议**：`_DETECTOR_LOCAL` 存一个 dict，按阈值缓存多个实例：
```python
_DETECTOR_LOCAL = threading.local()

def _get_detector(score_threshold=0.6):
    cache = getattr(_DETECTOR_LOCAL, "detectors", None)
    if cache is None:
        cache = _DETECTOR_LOCAL.detectors = {}
    if score_threshold not in cache:
        cache[score_threshold] = FaceDetector(score_threshold=score_threshold)
    return cache[score_threshold]
```

### Bug 10：landmark 子区域检测只取第一张脸

**位置**：`face_blur.py` per-face 循环 `sub_faces[0]`

**问题**：expand 后子区域可能包含相邻人脸（合影），只取 `sub_faces[0]` 的关键点可能张冠李戴。低概率，但多人合影场景存在。

**修复建议**：取子区域内**面积最大**的那张脸，或按与 face box 中心最近的一张：
```python
if sub_faces:
    f0 = max(sub_faces, key=lambda f: f["w"] * f["h"])
```

---

## 🟢 P3 — 代码质量（可选）

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| 11 | `app.py` L924 | L2 缓存命中不回填 L1，同 URL 反复查 DB | hit 时 `_cache_set(_ck, resp)` 回填 |
| 12 | `app.py` L99 | `DB_CACHE_TTL_HOURS` 定义未使用（死代码） | 删除，或让 `_db_cache_set` 用它 |
| 13 | `app.py` L978 | `/api/tasks/{id}` 返回裸 dict，admin 版返回 `{ok, task}`，格式不一致 | 统一为 `{ok: true, task: {...}}` |
| 14 | `face_blur.py` `_nms_faces` | `ordered.pop(0)` O(n²) | 人脸 >50 时改用 `collections.deque` |
| 15 | `face_blur.py` `handler()` | URL 下载无大小限制；`float(event.get(...))` 无保护 | 加 max_bytes + try/except |
| 16 | `app.py` 顶部 | 未使用 import：`io`、`shutil`、`cv2`、`numpy` | 清理 |
| 17 | `app.py` `healthz` | 每次触发模型加载检查 | 可接受，无需改 |

---

## 修复顺序建议

| 批次 | 内容 | 工作量 | 风险 |
|------|------|--------|------|
| **第一批（推荐立即）** | Bug 1（task_id 竞态）、Bug 4（SSRF） | 各 3-10 行 | 极低 |
| **第二批** | Bug 2（Session 线程安全）、Bug 3（缓存加锁）、Bug 6（模型下载锁） | 各 5-15 行 | 低 |
| **第三批** | Bug 5（并发语义）、Bug 8（insert 重试）、Bug 7（性能） | 中 | 低-中 |
| **按需** | Bug 9、10 及全部 P3 | — | 低 |

> 每次改动遵守部署规范：备份 → 改 → 验证 → 观察。
