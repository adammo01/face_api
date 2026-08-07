# 代码巡检 Bug 修改方案

> 基于 2026-08-07 验证后的修复计划
> 原则：每次改完一个批次立即测试，确认无误再继续

---

## 改动总览

| 批次 | Bug | 文件 | 行数 |
|------|-----|------|------|
| 第一批 | #1 task_id 竞态 | app.py | 2 行 |
| 第一批 | #2 Session 线程安全 | app.py | +12/-10 行 |
| 第一批 | #3 缓存加锁 | app.py | +6 行 |
| 第一批 | #4 SSRF | app.py | +25 行 |
| 第二批 | #6 模型下载锁 | face_blur.py | +5 行 |
| 第二批 | #11 L2 回填 L1 | app.py | +1 行 |
| 第二批 | #12 死代码 | app.py | -1 行 |
| 第三批 | #8 insert 重试 | app.py | +8 行 |
| 第三批 | #7 admin 双 stat | app.py | +3/-5 行 |

---

## 第一批（P0+P1，必修）

### Bug 1：L1 缓存 task_id 竞态

**文件**：`app.py` L733

**旧代码**：
```python
_cached["task_id"] = task_id
return _cached
```

**新代码**：
```python
return {**_cached, "task_id": task_id}
```

---

### Bug 2：Session 非线程安全

**文件**：`app.py` L544-554

**旧代码**：
```python
_DL_SESSION = None

def _dl_session():
    global _DL_SESSION
    if _DL_SESSION is None:
        import requests as _rq
        _DL_SESSION = _rq.Session()
        _adapter = _rq.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=50)
        _DL_SESSION.mount("https://", _adapter)
        _DL_SESSION.mount("http://", _adapter)
    return _DL_SESSION
```

**新代码**：
```python
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
```

说明：切换为 `threading.local()`，每线程独立 Session。连接池从 20/50 降到 5/10（每线程自己一份，总量相当）。

---

### Bug 3：L1 缓存加锁

**文件**：`app.py` `_cache_get()` + `_cache_set()`

在缓存模块开头（`_RESPONSE_CACHE` 后面）加锁：

**新增**：
```python
_CACHE_LOCK = threading.Lock()
```

**`_cache_get` 旧代码**：
```python
def _cache_get(key: str):
    if key not in _RESPONSE_CACHE:
        return None
    entry = _RESPONSE_CACHE[key]
    if time.time() - entry["ts"] > _CACHE_TTL_SECONDS:
        del _RESPONSE_CACHE[key]
        return None
    _RESPONSE_CACHE.move_to_end(key)
    return entry["resp"]
```

**`_cache_get` 新代码**：
```python
def _cache_get(key: str):
    with _CACHE_LOCK:
        if key not in _RESPONSE_CACHE:
            return None
        entry = _RESPONSE_CACHE[key]
        if time.time() - entry["ts"] > _CACHE_TTL_SECONDS:
            del _RESPONSE_CACHE[key]
            return None
        _RESPONSE_CACHE.move_to_end(key)
        return entry["resp"]
```

**`_cache_set` 旧代码**：
```python
def _cache_set(key: str, resp: dict):
    if key in _RESPONSE_CACHE:
        _RESPONSE_CACHE.move_to_end(key)
        _RESPONSE_CACHE[key] = {"ts": time.time(), "resp": resp}
    else:
        if len(_RESPONSE_CACHE) >= _CACHE_MAX_SIZE:
            _RESPONSE_CACHE.popitem(last=False)
        _RESPONSE_CACHE[key] = {"ts": time.time(), "resp": resp}
```

**`_cache_set` 新代码**：
```python
def _cache_set(key: str, resp: dict):
    with _CACHE_LOCK:
        if key in _RESPONSE_CACHE:
            _RESPONSE_CACHE.move_to_end(key)
            _RESPONSE_CACHE[key] = {"ts": time.time(), "resp": resp}
        else:
            if len(_RESPONSE_CACHE) >= _CACHE_MAX_SIZE:
                _RESPONSE_CACHE.popitem(last=False)
            _RESPONSE_CACHE[key] = {"ts": time.time(), "resp": resp}
```

---

### Bug 4：SSRF 防护

**文件**：`app.py` L556（`_download` 函数前）

**新增代码**（插在 `_download` 函数定义之前）：

```python
import ipaddress
import socket
import urllib.parse as _urlparse

# SSRF 防护: 禁止请求内网/保留地址
_BLOCKED_NETS = [
    ipaddress.IPv4Network(n) for n in (
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12",
        "192.168.0.0/16", "169.254.0.0/16", "0.0.0.0/8",
        "100.64.0.0/10", "198.18.0.0/15",
    )
] + [
    ipaddress.IPv6Network(n) for n in (
        "::1/128", "fc00::/7", "fe80::/10",
    )
]


def _assert_public_url(url: str):
    """检查 URL 目标是否为公网地址, 防止 SSRF"""
    try:
        host = _urlparse.urlparse(url).hostname
        if not host:
            raise HTTPException(400, "invalid URL: no hostname")
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            # hostname 不是 IP 字面量, DNS 解析
            info = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            ip = ipaddress.ip_address(info[0][4][0])
        if any(ip in net for net in _BLOCKED_NETS):
            raise HTTPException(400, f"blocked private/reserved address")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"URL validation failed: {e}")
```

在 `_download` 函数开头加校验：

```python
def _download(url: str, max_bytes: int = MAX_IMAGE_BYTES,
              timeout: int = DOWNLOAD_TIMEOUT) -> bytes:
    """下载 URL, 限制最大字节数 (复用连接池)"""
    _assert_public_url(url)          # ← 新增这一行
    sess = _dl_session()
    ...
```

---

## 第二批（P2，建议修）

### Bug 6：模型下载加锁

**文件**：`face_blur.py` L50

在 `_ensure_model` 前加锁：

```python
_MODEL_LOCK = threading.Lock()

def _ensure_model(model_url: str = YUNET_URL, model_path: Path = MODEL_PATH) -> Path:
    """确保模型文件存在, 缺失则下载."""
    if model_path.exists() and model_path.stat().st_size > 1000:
        return model_path
    with _MODEL_LOCK:
        # 双重检查: 拿到锁后再次确认（可能被其他线程刚下载完）
        if model_path.exists() and model_path.stat().st_size > 1000:
            return model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[face_blur] downloading model -> {model_path}")
        urllib.request.urlretrieve(model_url, model_path)
    return model_path
```

---

### Bug 11：L2 缓存命中回填 L1

**文件**：`app.py` L751（L2 缓存 return 之前）

在 `return {` 之前加一行：

```python
    if _db_url is not None:
        log.info("[cache] L2 hit")
        _cache_set(_ck, {                            # ← 新增
            "ok": True,
            "blocked": True,
            "face_count": -1,
            "elapsed_ms": 0,
            "process_ms": 0,
            "mode": req.mode,
            "output_url": _db_url,
            "original_url": str(req.image_url),
            "size": 0,
            "cached": True,
        })
        return {
```

---

### Bug 12：删除死代码

**文件**：`app.py` L99

删除这一行：

```python
DB_CACHE_TTL_HOURS = int(os.environ.get("FACE_BLUR_DB_CACHE_TTL_HOURS", str(IMAGE_TTL_HOURS)))
```

（`_db_cache_set` 实际用的是 `image_ttl_hours` 设置，此变量未引用）

---

## 第三批（P2 稳定性，择机修）

### Bug 8：_insert_request 重试

**文件**：`app.py` `_insert_request()`

**新代码**（替换原函数体）：

```python
def _insert_request(row: dict) -> None:
    for attempt in range(3):
        try:
            with _db() as conn:
                conn.execute(
                    """
                    INSERT INTO requests (
                        task_id, created_at, status, mode, blocked, face_count, elapsed_ms,
                        process_ms, input_bytes, output_bytes, image_url, output_url,
                        output_file, error, client_ip, user_agent, attempts, retried,
                        request_json, response_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("task_id"),
                        row.get("created_at", _utc_now()),
                        row.get("status", "unknown"),
                        row.get("mode"),
                        int(row.get("blocked", 0)),
                        int(row.get("face_count", 0)),
                        float(row.get("elapsed_ms", 0)),
                        float(row.get("process_ms", 0)),
                        int(row.get("input_bytes", 0)),
                        int(row.get("output_bytes", 0)),
                        row.get("image_url"),
                        row.get("output_url"),
                        row.get("output_file"),
                        row.get("error"),
                        row.get("client_ip"),
                        row.get("user_agent"),
                        int(row.get("attempts", 1)),
                        int(row.get("retried", 0)),
                        row.get("request_json"),
                        row.get("response_json"),
                    ),
                )
            return  # success
        except Exception as e:
            msg = str(e)
            if "locked" in msg.lower() and attempt < 2:
                time.sleep(0.05 * (attempt + 1))
                continue
            log.warning("request log insert failed: %s", e)
            return
```

---

### Bug 7：admin 双 stat 优化

**文件**：`app.py` `_static_files()` L424-439

**新代码**：

```python
def _static_files(offset: int = 0, limit: int | None = None) -> tuple[list[dict], int]:
    # 一次 stat 存入元组, 避免排序和循环各 stat 一次
    entries = [(p, p.stat()) for p in STATIC_DIR.glob("*.jpg")]
    entries.sort(key=lambda x: x[1].st_mtime, reverse=True)
    total = len(entries)
    if limit is not None:
        entries = entries[offset:offset + limit]
    else:
        entries = entries[offset:]
    files = []
    for path, st in entries:
        files.append({
            "name": path.name,
            "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "url": f"/static/{path.name}",
        })
    return files, total
```

---

## 不修的项目及原因

| Bug | 原因 |
|-----|------|
| #5 并发每进程 | 当前 QPS 极低 (~2/min)，512 上限远未触及。Admin 面板加提示即可，不改代码 |
| #9 detector 重建 | 当前业务统一用 0.45，不存在阈值抖动，无需缓存多实例 |
| #10 sub 第一张脸 | 触发概率极低（expand 后恰好包含相邻人脸），影响小 |
| #13 task_status 格式 | 客户端已适配两种格式，贸改可能破坏兼容性 |
| #14 NMS O(n²) | 人脸数 <10，无影响 |
| #15 handler 下载 | Serverless 路径未实际部署 |
| #16 冗余 import | 不影响运行，清理可能引入 import 顺序问题 |

---

## 执行顺序

```
1. 备份 → /root/faceblur_backup_fix_batch1
2. 第一批: Bug 1,2,3,4 (app.py 共 ~50 行改动)
3. 语法检查 + 上传 + 重启 + healthz 验证
4. 备份 → /root/faceblur_backup_fix_batch2
5. 第二批: Bug 6,11,12 (face_blur.py + app.py)
6. 语法检查 + 上传 + 重启 + healthz 验证
7. 第三批: Bug 8,7 (按需)
```
