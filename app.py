"""
face_blur API 服务
==================

启动:
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2

接口:
    POST /api/face_blur
        body: {
            "image_url": "https://...",     # 必填, 公网可访问的图片 URL
            "mode": "pixelate | gaussian | solid | landmark | landmark_whole_face",  # 默认 gaussian
            "score_threshold": 0.45,        # 可选, 隐私打码默认偏召回
            "expand_ratio": 0.35,           # 可选, 扩大覆盖脸部边缘
            "dot_radius": 4,                # landmark / landmark_whole_face 模式专用
            "spacing": 12,                  # landmark / landmark_whole_face 模式专用
            "face_grid_step": 14,           # landmark_whole_face 模式专用, 整脸网格步长
            "grid_n": 5,                    # landmark_whole_face 模式专用, 关键点附近矩阵大小
            "callback_url": "https://..."   # 可选, 完成后回调
        }
        resp: {
            "ok": true,
            "face_count": 3,
            "elapsed_ms": 124.5,
            "mode": "gaussian",
            "blocked": true,                # 是否做了打码
            "original_url": "...",          # 输入图原 URL
            "output_url": "..."             # 打码后图的公网 URL (blocked=false 时 == original_url)
        }

新增 mode "landmark_whole_face" (2026-08-05):
  - 整脸范围均匀打红点 (face_grid_step) + 关键点附近 grid_n×grid_n 密集叠加
  - 推荐参数: face_grid_step=14, dot_radius=3, grid_n=5, expand_ratio=0.30
  - 实测: 多图 reference_image 场景下能过火山方舟 InputImageSensitiveContentDetected

    GET /healthz
        resp: {"ok": true, "model_loaded": true, "pid": 12345}

    GET /static/{filename}
        返回打码后的静态图(JPEG)
"""
from __future__ import annotations

import os
import sys
import time
import uuid
import shutil
import hashlib
import logging
import json
import sqlite3
import threading
import ipaddress
import socket as _socket_mod
import urllib.parse as _urlparse_mod
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

# 加载本地 face_blur 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_blur import process_image  # noqa: E402

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

STATIC_DIR = Path(os.environ.get("FACE_BLUR_STATIC_DIR",
                                  Path(__file__).resolve().parent / "static"))
STATIC_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(os.environ.get("FACE_BLUR_DATA_DIR",
                                Path(__file__).resolve().parent / "logs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.environ.get("FACE_BLUR_DB_PATH", DATA_DIR / "faceblur.sqlite3"))

# 公网访问 base URL, 用于生成 output_url
# - 反向代理模式: 设成 "https://your-domain.com" (Nginx + HTTPS)
# - 直接暴露模式: 设成 "http://<server-ip>:8000"
# - 留空则用 request.base_url
PUBLIC_BASE_URL = os.environ.get("FACE_BLUR_PUBLIC_URL", "")

# 鉴权 (可选, 用法见 README)
API_TOKEN = os.environ.get("FACE_BLUR_API_TOKEN", "")
ADMIN_TOKEN = os.environ.get("FACE_BLUR_ADMIN_TOKEN", API_TOKEN)

# 下载图时的超时和最大尺寸
DOWNLOAD_TIMEOUT = int(os.environ.get("FACE_BLUR_DL_TIMEOUT", "30"))
MAX_IMAGE_BYTES = int(os.environ.get("FACE_BLUR_MAX_BYTES", str(20 * 1024 * 1024)))  # 20 MB
IMAGE_TTL_HOURS = int(os.environ.get("FACE_BLUR_IMAGE_TTL_HOURS", "72"))
DB_CACHE_TTL_HOURS = int(os.environ.get("FACE_BLUR_DB_CACHE_TTL_HOURS", str(IMAGE_TTL_HOURS)))
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("FACE_BLUR_CLEANUP_INTERVAL_SECONDS", "3600"))
MAX_RETRIES = int(os.environ.get("FACE_BLUR_MAX_RETRIES", "2"))
RETRY_BACKOFF_SECONDS = float(os.environ.get("FACE_BLUR_RETRY_BACKOFF_SECONDS", "0.6"))
DEFAULT_MAX_CONCURRENT_TASKS = int(os.environ.get("FACE_BLUR_MAX_CONCURRENT_TASKS", "4"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("faceblur-api")

_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT_TASKS = 0

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="face_blur API", version="1.1.0",
              description="人脸打码服务 (pixelate/gaussian/solid/landmark/landmark_whole_face)")

# 静态文件服务(返回打码后的图)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db() -> None:
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                mode TEXT,
                blocked INTEGER NOT NULL DEFAULT 0,
                face_count INTEGER NOT NULL DEFAULT 0,
                elapsed_ms REAL NOT NULL DEFAULT 0,
                process_ms REAL NOT NULL DEFAULT 0,
                input_bytes INTEGER NOT NULL DEFAULT 0,
                output_bytes INTEGER NOT NULL DEFAULT 0,
                image_url TEXT,
                output_url TEXT,
                output_file TEXT,
                error TEXT,
                client_ip TEXT,
                user_agent TEXT,
                attempts INTEGER NOT NULL DEFAULT 1,
                retried INTEGER NOT NULL DEFAULT 0,
                request_json TEXT,
                response_json TEXT
            )
            """
        )
        existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
        if "attempts" not in existing_cols:
            conn.execute("ALTER TABLE requests ADD COLUMN attempts INTEGER NOT NULL DEFAULT 1")
        if "retried" not in existing_cols:
            conn.execute("ALTER TABLE requests ADD COLUMN retried INTEGER NOT NULL DEFAULT 0")
        if "request_json" not in existing_cols:
            conn.execute("ALTER TABLE requests ADD COLUMN request_json TEXT")
        if "parent_task_id" not in existing_cols:
            conn.execute("ALTER TABLE requests ADD COLUMN parent_task_id TEXT")
        if "task_id" not in existing_cols:
            conn.execute("ALTER TABLE requests ADD COLUMN task_id TEXT")
        if "response_json" not in existing_cols:
            conn.execute("ALTER TABLE requests ADD COLUMN response_json TEXT")
        conn.execute(
            "UPDATE requests SET task_id = printf('legacy-%012d', id) WHERE task_id IS NULL"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_requests_task_id ON requests(task_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cleanup_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ttl_hours INTEGER NOT NULL,
                deleted_files INTEGER NOT NULL,
                freed_bytes INTEGER NOT NULL,
                error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        defaults = {
            "max_concurrent_tasks": str(DEFAULT_MAX_CONCURRENT_TASKS),
            "max_retries": str(MAX_RETRIES),
            "retry_backoff_seconds": str(RETRY_BACKOFF_SECONDS),
            "image_ttl_hours": str(IMAGE_TTL_HOURS),
        }
        for key, value in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, _utc_now()),
            )
        # DB 持久缓存: 原始URL -> 打码后URL 映射
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blur_cache (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key   TEXT UNIQUE NOT NULL,
                image_url   TEXT NOT NULL,
                output_url  TEXT NOT NULL,
                output_file TEXT NOT NULL,
                mode        TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                expires_at  TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blur_cache_key ON blur_cache(cache_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blur_cache_expires ON blur_cache(expires_at)"
        )


def _get_setting(key: str, default: str) -> str:
    try:
        with _db() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default
    except Exception as e:  # noqa: BLE001
        log.warning("settings read failed: %s", e)
        return default


def _get_int_setting(key: str, default: int, min_value: int = 1, max_value: int = 100) -> int:
    try:
        value = int(float(_get_setting(key, str(default))))
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _get_float_setting(key: str, default: float, min_value: float = 0.0, max_value: float = 60.0) -> float:
    try:
        value = float(_get_setting(key, str(default)))
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _set_settings(values: dict) -> dict:
    allowed = {
        "max_concurrent_tasks": (1, 128, int),
        "max_retries": (0, 10, int),
        "retry_backoff_seconds": (0.0, 10.0, float),
        "image_ttl_hours": (1, 24 * 365, int),
    }
    saved = {}
    with _db() as conn:
        for key, raw in values.items():
            if key not in allowed:
                continue
            lo, hi, caster = allowed[key]
            val = caster(raw)
            val = max(lo, min(hi, val))
            text = str(val)
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, text, _utc_now()),
            )
            saved[key] = val
    return saved


@contextmanager
def _task_slot():
    global _INFLIGHT_TASKS
    limit = _get_int_setting("max_concurrent_tasks", DEFAULT_MAX_CONCURRENT_TASKS, 1, 128)
    with _INFLIGHT_LOCK:
        if _INFLIGHT_TASKS >= limit:
            raise HTTPException(429, f"too many concurrent tasks ({_INFLIGHT_TASKS}/{limit})")
        _INFLIGHT_TASKS += 1
    try:
        yield limit
    finally:
        with _INFLIGHT_LOCK:
            _INFLIGHT_TASKS = max(0, _INFLIGHT_TASKS - 1)


def _insert_request(row: dict) -> None:
    for _attempt in range(3):
        try:
            with _db() as conn:
                conn.execute(
                    """
                    INSERT INTO requests (
                        task_id, created_at, status, mode, blocked, face_count, elapsed_ms,
                        process_ms, input_bytes, output_bytes, image_url, output_url,
                        output_file, parent_task_id, error, client_ip, user_agent, attempts, retried,
                        request_json, response_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        row.get("parent_task_id"),
                        row.get("error"),
                        row.get("client_ip"),
                        row.get("user_agent"),
                        int(row.get("attempts", 1)),
                        int(row.get("retried", 0)),
                        row.get("request_json"),
                        row.get("response_json"),
                    ),
                )
            return
        except Exception as e:
            msg = str(e)
            if "locked" in msg.lower() and _attempt < 2:
                time.sleep(0.05 * (_attempt + 1))
                continue
            log.warning("request log insert failed: %s", e)
            return


def _require_admin(
    request: Request,
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> None:
    if not ADMIN_TOKEN:
        client = request.client.host if request.client else ""
        if client in {"127.0.0.1", "::1", "localhost"}:
            return
        raise HTTPException(403, "admin disabled: set FACE_BLUR_ADMIN_TOKEN")

    presented = x_admin_token or token or ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization.split(" ", 1)[1].strip()
    if presented != ADMIN_TOKEN:
        raise HTTPException(401, "invalid admin token")


_SENSITIVE_HEADERS = {
    "authorization", "cookie", "proxy-authorization", "set-cookie",
    "cf-access-jwt-assertion", "x-api-key", "x-admin-token", "x-auth-token",
}


def _is_sensitive_header(name: str) -> bool:
    lowered = name.lower()
    return lowered in _SENSITIVE_HEADERS or any(
        marker in lowered for marker in ("token", "secret", "api-key", "apikey")
    )


def _request_metadata(request: Request) -> tuple[dict, dict]:
    headers = {
        key.lower(): ("[REDACTED]" if _is_sensitive_header(key) else value)
        for key, value in request.headers.items()
    }
    query: dict[str, str | list[str]] = {}
    for key, value in request.query_params.multi_items():
        if key in query:
            current = query[key]
            query[key] = [*current, value] if isinstance(current, list) else [current, value]
        else:
            query[key] = value
    return headers, query


def _request_record(req: "FaceBlurRequest", request: Request) -> str:
    headers, query = _request_metadata(request)
    return json.dumps({
        "method": request.method,
        "path": request.url.path,
        "query": query,
        "headers": headers,
        "body": req.model_dump(mode="json"),
    }, ensure_ascii=False)


def _raw_request_record(request: Request, body) -> str:
    headers, query = _request_metadata(request)
    return json.dumps({
        "method": request.method,
        "path": request.url.path,
        "query": query,
        "headers": headers,
        "body": body,
    }, ensure_ascii=False)


def _decode_json(raw: str | None):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _task_row(row: sqlite3.Row | dict) -> dict:
    item = dict(row)
    item["request"] = _decode_json(item.pop("request_json", None))
    item["response"] = _decode_json(item.pop("response_json", None))
    return item


def _static_files(offset: int = 0, limit: int | None = None,
                  include_task_id: bool = False) -> tuple[list[dict], int]:
    paths = sorted(STATIC_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    total = len(paths)
    if limit is not None:
        paths = paths[offset:offset + limit]
    else:
        paths = paths[offset:]
    # Batch lookup task_id for image click navigation
    task_map = {}
    if include_task_id and paths:
        try:
            file_names = [p.name for p in paths]
            with _db() as conn:
                placeholders = ",".join("?" for _ in file_names)
                rows = conn.execute(
                    f"SELECT output_file, task_id FROM requests WHERE output_file IN ({placeholders})",
                    file_names,
                ).fetchall()
                task_map = {r["output_file"]: r["task_id"] for r in rows}
        except Exception as e:
            log.warning("task_id lookup failed: %s", e)
    files = []
    for path in paths:
        st = path.stat()
        item = {
            "name": path.name,
            "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "url": f"/static/{path.name}",
        }
        if include_task_id:
            item["task_id"] = task_map.get(path.name, "")
        files.append(item)
    return files, total


def _cleanup_static(ttl_hours: int = IMAGE_TTL_HOURS) -> dict:
    cutoff = time.time() - ttl_hours * 3600
    deleted = 0
    freed = 0
    error = None
    try:
        for path in STATIC_DIR.glob("*.jpg"):
            try:
                st = path.stat()
                if st.st_mtime >= cutoff:
                    continue
                freed += st.st_size
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                continue
        return {"deleted_files": deleted, "freed_bytes": freed, "ttl_hours": ttl_hours}
    except Exception as e:  # noqa: BLE001
        error = str(e)
        raise
    finally:
        try:
            with _db() as conn:
                conn.execute(
                    "INSERT INTO cleanup_runs (created_at, ttl_hours, deleted_files, freed_bytes, error) VALUES (?, ?, ?, ?, ?)",
                    (_utc_now(), ttl_hours, deleted, freed, error),
                )
        except Exception as e:  # noqa: BLE001
            log.warning("cleanup log insert failed: %s", e)


def _cleanup_loop() -> None:
    while True:
        time.sleep(max(60, CLEANUP_INTERVAL_SECONDS))
        try:
            result = _cleanup_static(_get_int_setting("image_ttl_hours", IMAGE_TTL_HOURS, 1, 24 * 365))
            if result["deleted_files"]:
                log.info("cleanup deleted=%s freed=%s", result["deleted_files"], result["freed_bytes"])
        except Exception as e:  # noqa: BLE001
            log.warning("cleanup failed: %s", e)
        try:
            _db_cache_cleanup_expired()
        except Exception as e:
            log.warning("db_cache_cleanup failed: %s", e)


@app.on_event("startup")
def startup() -> None:
    _init_db()
    thread = threading.Thread(target=_cleanup_loop, daemon=True, name="faceblur-cleanup")
    thread.start()


class FaceBlurRequest(BaseModel):
    image_url: HttpUrl
    mode: str = Field(
        "gaussian",
        pattern=r"^(pixelate|gaussian|solid|landmark|landmark_whole_face)$",
    )
    # 默认值已对齐 face_blur.py / v14r3 推荐参数 (2026-08-05)
    score_threshold: float = Field(0.52, ge=0.1, le=0.99)
    expand_ratio: float = Field(0.30, ge=0.0, le=1.0)
    dot_radius: int = Field(3, ge=1, le=20)
    spacing: int = Field(14, ge=4, le=60)
    # landmark_whole_face 模式专用参数
    face_grid_step: int = Field(14, ge=4, le=60)
    grid_n: int = Field(5, ge=3, le=11, description="关键点附近矩阵大小 (建议 3-7)")
    parent_task_id: Optional[str] = Field(None, max_length=200, description="上游任务批次标记")
    callback_url: Optional[HttpUrl] = None


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    if request.url.path != "/api/face_blur":
        return JSONResponse({"detail": exc.errors()}, status_code=422)
    task_id = uuid.uuid4().hex
    raw = await request.body()
    try:
        body = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        body = raw.decode("utf-8", errors="replace")
    error = "request validation failed"
    response = {"task_id": task_id, "status": "validation_error", "error": error, "errors": exc.errors()}
    _insert_request({
        "task_id": task_id,
        "status": "validation_error",
        "mode": body.get("mode") if isinstance(body, dict) else None,
        "image_url": body.get("image_url") if isinstance(body, dict) else None,
        "error": error,
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent", "")[:300],
        "request_json": _raw_request_record(request, body),
        "response_json": json.dumps(response, ensure_ascii=False),
    })
    return JSONResponse(response, status_code=422)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

# SSRF 防护: 禁止请求内网/保留地址
_BLOCKED_NETS = []
for _n in ("127.0.0.0/8","10.0.0.0/8","172.16.0.0/12","192.168.0.0/16","169.254.0.0/16","0.0.0.0/8","100.64.0.0/10","198.18.0.0/15"):
    _BLOCKED_NETS.append(ipaddress.IPv4Network(_n))
for _n in ("::1/128","fc00::/7","fe80::/10"):
    _BLOCKED_NETS.append(ipaddress.IPv6Network(_n))

def _assert_public_url(url: str):
    host = _urlparse_mod.urlparse(url).hostname
    if not host:
        raise HTTPException(400, "invalid URL: no hostname")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        info = _socket_mod.getaddrinfo(host, None, type=_socket_mod.SOCK_STREAM)
        ip = ipaddress.ip_address(info[0][4][0])
    check_ips = [ip]
    if ip.version == 6 and ip.ipv4_mapped:
        check_ips.append(ip.ipv4_mapped)
    if any(any(cip in net for net in _BLOCKED_NETS) for cip in check_ips):
        raise HTTPException(400, "blocked private/reserved address")

# C: requests.Session 连接池复用 (thread-local, 线程安全)
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

def _download(url: str, max_bytes: int = MAX_IMAGE_BYTES,
              timeout: int = DOWNLOAD_TIMEOUT) -> bytes:
    """下载 URL, 限制最大字节数 (复用连接池)"""
    _assert_public_url(url)
    sess = _dl_session()
    with sess.get(url, timeout=(5, timeout), stream=True,
                  headers={"User-Agent": "faceblur-api/1.0"}) as resp:
        resp.raise_for_status()
        data = b""
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                break
            data += chunk
            if len(data) > max_bytes:
                raise HTTPException(413, f"image too large (> {max_bytes} bytes)")
    return data


def _run_with_retries(label: str, func, max_retries: int = MAX_RETRIES):
    """Run a transient operation with bounded retries and exponential backoff."""
    attempts = 0
    last_error: Exception | None = None
    for attempt in range(max(0, max_retries) + 1):
        attempts = attempt + 1
        try:
            return func(), attempts, None
        except HTTPException:
            raise
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt >= max_retries:
                break
            backoff = _get_float_setting("retry_backoff_seconds", RETRY_BACKOFF_SECONDS, 0.0, 10.0)
            sleep_s = backoff * (2 ** attempt)
            log.warning("%s attempt %s failed: %s; retrying in %.1fs", label, attempts, e, sleep_s)
            time.sleep(sleep_s)
    assert last_error is not None
    return None, attempts, last_error



# E: URL 级 LRU 缓存 (TTL 5分钟, 最多 200 条)
import hashlib
from collections import OrderedDict

_RESPONSE_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX_SIZE = 200
_CACHE_TTL_SECONDS = 300
_CACHE_LOCK = threading.Lock()

def _cache_key(image_url: str, req_json: str) -> str:
    return hashlib.md5((image_url + req_json).encode()).hexdigest()

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

def _cache_set(key: str, resp: dict):
    with _CACHE_LOCK:
        if key in _RESPONSE_CACHE:
            _RESPONSE_CACHE.move_to_end(key)
            _RESPONSE_CACHE[key] = {"ts": time.time(), "resp": resp}
        else:
            if len(_RESPONSE_CACHE) >= _CACHE_MAX_SIZE:
                _RESPONSE_CACHE.popitem(last=False)
            _RESPONSE_CACHE[key] = {"ts": time.time(), "resp": resp}

# DB 持久缓存 (L2): 原始URL->打码后URL 映射, 进程重启不丢
def _db_cache_get(cache_key: str):
    """查询 L2 缓存。返回 output_url 或 None。过期/文件丢失自动清理。"""
    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT output_url, output_file, expires_at FROM blur_cache WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if row["expires_at"] < now:
            _db_cache_del(cache_key)
            return None
        out_path = STATIC_DIR / row["output_file"]
        if not out_path.exists():
            _db_cache_del(cache_key)
            return None
        return row["output_url"]
    except Exception as e:
        log.warning("db_cache_get failed: %s", e)
        return None


def _db_cache_set(cache_key: str, image_url: str, output_url: str,
                  output_file: str, mode: str):
    """写入 L2 缓存"""
    try:
        now = _utc_now()
        ttl_hours = _get_int_setting("image_ttl_hours", IMAGE_TTL_HOURS, 1, 24 * 365)
        expires = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat(timespec="seconds")
        with _db() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO blur_cache
                   (cache_key, image_url, output_url, output_file, mode, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (cache_key, image_url, output_url, output_file, mode, now, expires),
            )
    except Exception as e:
        log.warning("db_cache_set failed: %s", e)


def _db_cache_del(cache_key: str):
    """删除 L2 缓存记录"""
    try:
        with _db() as conn:
            conn.execute("DELETE FROM blur_cache WHERE cache_key=?", (cache_key,))
    except Exception as e:
        log.warning("db_cache_del failed: %s", e)


def _db_cache_cleanup_expired():
    """清理所有过期的 L2 缓存记录"""
    try:
        now = _utc_now()
        with _db() as conn:
            result = conn.execute(
                "DELETE FROM blur_cache WHERE expires_at < ?", (now,)
            )
        deleted = result.rowcount
        if deleted:
            log.info("db_cache_cleanup: deleted %d expired records", deleted)
        return deleted
    except Exception as e:
        log.warning("db_cache_cleanup failed: %s", e)
        return 0


def _public_url_for(path: Path, request_base: str) -> str:
    """生成 output_url. 优先用 PUBLIC_BASE_URL, 否则用 request base_url"""
    rel = path.name
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL.rstrip('/')}/static/{rel}"
    return f"{request_base}static/{rel}"


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    # 触发模型加载检查
    try:
        from face_blur import _get_detector
        det = _get_detector()
        model_loaded = det is not None
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "model_loaded": False, "error": str(e)},
                            status_code=500)
    return {"ok": True, "model_loaded": model_loaded, "pid": os.getpid()}


@app.post("/api/face_blur")
def face_blur(req: FaceBlurRequest, request: Request):
    task_id = uuid.uuid4().hex
    # E: 计算缓存 key (L1 + L2 共用)
    _cache_req_json = json.dumps(req.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    _ck = _cache_key(str(req.image_url), _cache_req_json)

    # L1: 内存缓存
    _cached = _cache_get(_ck)
    if _cached is not None:
        log.info("[cache] L1 hit")
        return {**_cached, "task_id": task_id, "parent_task_id": req.parent_task_id or ""}

    # L2: DB 持久缓存
    _db_url = _db_cache_get(_ck)
    if _db_url is not None:
        # 构造与正常打码一致的响应
        log.info("[cache] L2 hit")
        return {
            "task_id": task_id,
            "ok": True,
            "blocked": True,
            "face_count": -1,
            "elapsed_ms": 0,
            "mode": req.mode,
            "output_url": _db_url,
            "original_url": str(req.image_url),
            "parent_task_id": req.parent_task_id or "",
            "cached": True,
        }

    try:
        with _task_slot():
            return _face_blur_impl(task_id, req, request, cache_key=_ck)
    except HTTPException as exc:
        response = {"task_id": task_id, "status": "rejected", "error": str(exc.detail)}
        _insert_request({
            "task_id": task_id,
            "status": "rejected",
            "mode": req.mode,
            "image_url": str(req.image_url),
            "error": str(exc.detail),
            "parent_task_id": req.parent_task_id or "",
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent", "")[:300],
            "request_json": _request_record(req, request),
            "response_json": json.dumps(response, ensure_ascii=False),
        })
        return JSONResponse(response, status_code=exc.status_code)


def _face_blur_impl(task_id: str, req: FaceBlurRequest, request: Request, *, cache_key: str = ""):
    t0 = time.perf_counter()
    image_url = str(req.image_url)
    request_json = _request_record(req, request)
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent", "")[:300]
    attempts = 1
    max_retries = _get_int_setting("max_retries", MAX_RETRIES, 0, 10)

    log.info(f"[req] mode={req.mode}  url={image_url[:120]}...")

    # 1. 下载
    try:
        img_bytes, download_attempts, download_error = _run_with_retries(
            "download", lambda: _download(image_url), max_retries=max_retries
        )
        attempts = max(attempts, download_attempts)
        if download_error is not None:
            raise download_error
    except HTTPException as e:
        response = {"task_id": task_id, "status": "download_error", "error": str(e.detail)}
        _insert_request({
            "task_id": task_id, "status": "download_error", "mode": req.mode,
            "image_url": image_url, "error": str(e.detail),
            "parent_task_id": req.parent_task_id or "",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "client_ip": client_ip, "user_agent": user_agent,
            "attempts": attempts, "retried": attempts > 1,
            "request_json": request_json,
            "response_json": json.dumps(response, ensure_ascii=False),
        })
        return JSONResponse(response, status_code=e.status_code)
    except Exception as e:
        log.error(f"download failed: {e}")
        response = {"task_id": task_id, "status": "download_error", "error": str(e)}
        _insert_request({
            "task_id": task_id,
            "status": "download_error",
            "mode": req.mode,
            "image_url": image_url,
            "parent_task_id": req.parent_task_id or "",
            "error": str(e),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "attempts": attempts,
            "retried": attempts > 1,
            "request_json": request_json,
            "response_json": json.dumps(response, ensure_ascii=False),
        })
        return JSONResponse(response, status_code=400)
    log.info(f"  downloaded {len(img_bytes)} bytes  ({int((time.perf_counter()-t0)*1000)}ms)")

    # 2. 打码
    try:
        blur_params: dict = {}
        if req.mode == "landmark":
            blur_params["dot_radius"] = req.dot_radius
            blur_params["spacing"] = req.spacing
        elif req.mode == "landmark_whole_face":
            blur_params["dot_radius"] = req.dot_radius
            blur_params["spacing"] = req.spacing
            blur_params["face_grid_step"] = req.face_grid_step
            blur_params["grid_n"] = req.grid_n
        result, process_attempts, process_error = _run_with_retries(
            "process_image",
            lambda: process_image(
                img_bytes,
                mode=req.mode,
                score_threshold=req.score_threshold,
                expand_ratio=req.expand_ratio,
                return_faces=True,
                **blur_params,
            ),
            max_retries=max_retries,
        )
        attempts += process_attempts - 1
        if process_error is not None:
            raise process_error
    except ValueError as e:
        response = {"task_id": task_id, "status": "bad_request", "error": str(e)}
        _insert_request({
            "task_id": task_id,
            "status": "bad_request",
            "mode": req.mode,
            "image_url": image_url,
            "input_bytes": len(img_bytes),
            "parent_task_id": req.parent_task_id or "",
            "error": str(e),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "attempts": attempts,
            "retried": attempts > 1,
            "request_json": request_json,
            "response_json": json.dumps(response, ensure_ascii=False),
        })
        return JSONResponse(response, status_code=400)
    except Exception as e:  # noqa: BLE001
        log.exception("process_image failed")
        response = {"task_id": task_id, "status": "process_error", "error": str(e)}
        _insert_request({
            "task_id": task_id,
            "status": "process_error",
            "mode": req.mode,
            "image_url": image_url,
            "input_bytes": len(img_bytes),
            "parent_task_id": req.parent_task_id or "",
            "error": str(e),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "attempts": attempts,
            "retried": attempts > 1,
            "request_json": request_json,
            "response_json": json.dumps(response, ensure_ascii=False),
        })
        return JSONResponse(response, status_code=500)

    face_count = result["face_count"]
    blocked = face_count > 0
    elapsed = round(result["elapsed_ms"], 2)
    log.info(f"  detected {face_count} faces  blocked={blocked}  ({elapsed}ms)")

    # 3. 没检测到人脸: 直接 echo 原 URL, 不下载/上传
    if not blocked:
        response = {
            "task_id": task_id,
            "ok": True,
            "blocked": False,
            "face_count": 0,
            "elapsed_ms": elapsed,
            "mode": req.mode,
            "original_url": image_url,
            "output_url": image_url,
            "message": "no face detected, return original url",
        }
        _insert_request({
            "task_id": task_id,
            "status": "ok",
            "mode": req.mode,
            "blocked": False,
            "face_count": 0,
            "elapsed_ms": elapsed,
            "process_ms": elapsed,
            "input_bytes": len(img_bytes),
            "image_url": image_url,
            "output_url": image_url,
            "parent_task_id": req.parent_task_id or "",
            "client_ip": client_ip,
            "user_agent": user_agent,
            "attempts": attempts,
            "retried": attempts > 1,
            "request_json": request_json,
            "response_json": json.dumps(response, ensure_ascii=False),
        })
        _cache_set(cache_key, response)
        return response

    # 4. 写入静态目录
    out_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
    out_path = STATIC_DIR / out_name
    out_path.write_bytes(result["image_bytes"])
    log.info(f"  saved {out_path}  ({len(result['image_bytes'])} bytes)")

    # 5. 生成 output_url
    request_base = str(request.base_url)
    output_url = _public_url_for(out_path, request_base)

    elapsed_total = round((time.perf_counter() - t0) * 1000, 2)
    response = {
        "task_id": task_id,
        "ok": True,
        "blocked": True,
        "face_count": face_count,
        "elapsed_ms": elapsed_total,
        "process_ms": elapsed,
        "mode": req.mode,
        "original_url": image_url,
        "output_url": output_url,
        "parent_task_id": req.parent_task_id or "",
        "size": len(result["image_bytes"]),
    }
    _insert_request({
        "task_id": task_id,
        "status": "ok",
        "mode": req.mode,
        "blocked": True,
        "face_count": face_count,
        "elapsed_ms": elapsed_total,
        "process_ms": elapsed,
        "input_bytes": len(img_bytes),
        "output_bytes": len(result["image_bytes"]),
        "image_url": image_url,
        "output_url": output_url,
        "output_file": out_name,
        "parent_task_id": req.parent_task_id or "",
        "client_ip": client_ip,
        "user_agent": user_agent,
        "attempts": attempts,
        "retried": attempts > 1,
        "request_json": request_json,
        "response_json": json.dumps(response, ensure_ascii=False),
    })

    # L2: 写入 DB 持久缓存
    _db_cache_set(cache_key, str(req.image_url), response["output_url"],
                  out_name, req.mode)
    _cache_set(cache_key, response)
    return response


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str):
    with _db() as conn:
        row = conn.execute(
            """
            SELECT task_id, created_at, status, mode, blocked, face_count, elapsed_ms,
                   process_ms, output_url, error, attempts, retried
            FROM requests WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "task not found")
    return dict(row)


@app.get("/api/admin/summary")
def admin_summary(
    request: Request,
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_admin(request, authorization, x_admin_token, token)
    with _db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        ok = conn.execute("SELECT COUNT(*) FROM requests WHERE status='ok'").fetchone()[0]
        blocked = conn.execute("SELECT COUNT(*) FROM requests WHERE blocked=1").fetchone()[0]
        errors = conn.execute("SELECT COUNT(*) FROM requests WHERE status!='ok'").fetchone()[0]
        today = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE created_at >= datetime('now', '-24 hours')"
        ).fetchone()[0]
        today_ok = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE status='ok' AND created_at >= datetime('now', '-24 hours')"
        ).fetchone()[0]
        retried = conn.execute("SELECT COUNT(*) FROM requests WHERE retried=1").fetchone()[0]
        avg_ms = conn.execute(
            "SELECT COALESCE(AVG(elapsed_ms), 0) FROM requests WHERE status='ok'"
        ).fetchone()[0]
        by_mode = [dict(r) for r in conn.execute(
            "SELECT mode, COUNT(*) AS count FROM requests GROUP BY mode ORDER BY count DESC"
        )]
        cleanup = conn.execute(
            "SELECT * FROM cleanup_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    files, file_total = _static_files()
    with _INFLIGHT_LOCK:
        inflight = _INFLIGHT_TASKS
    return {
        "ok": True,
        "service": {
            "pid": os.getpid(),
            "static_dir": str(STATIC_DIR),
            "db_path": str(DB_PATH),
            "public_base_url": PUBLIC_BASE_URL,
            "image_ttl_hours": IMAGE_TTL_HOURS,
            "cleanup_interval_seconds": CLEANUP_INTERVAL_SECONDS,
            "max_retries": _get_int_setting("max_retries", MAX_RETRIES, 0, 10),
            "retry_backoff_seconds": _get_float_setting("retry_backoff_seconds", RETRY_BACKOFF_SECONDS, 0.0, 10.0),
            "image_ttl_hours_effective": _get_int_setting("image_ttl_hours", IMAGE_TTL_HOURS, 1, 24 * 365),
            "inflight_tasks": inflight,
            "max_concurrent_tasks": _get_int_setting("max_concurrent_tasks", DEFAULT_MAX_CONCURRENT_TASKS, 1, 128),
        },
        "requests": {
            "total": total,
            "ok": ok,
            "blocked": blocked,
            "errors": errors,
            "last_24h": today,
            "last_24h_ok": today_ok,
            "success_rate": round((ok / total * 100) if total else 0, 2),
            "success_rate_24h": round((today_ok / today * 100) if today else 0, 2),
            "retried": retried,
            "avg_elapsed_ms": round(float(avg_ms), 2),
            "by_mode": by_mode,
        },
        "storage": {
            "file_count": file_total,
            "bytes": sum(f["size"] for f in files),
        },
        "last_cleanup": dict(cleanup) if cleanup else None,
    }


@app.get("/api/admin/requests")
def admin_requests(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(10, ge=10, le=200),
    status: str | None = Query(default=None),
    parent_task_id: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_admin(request, authorization, x_admin_token, token)
    where = ""
    params: list = []
    if status:
        where = " WHERE status = ?"
        params.append(status)
    if parent_task_id:
        prefix = " AND " if where else " WHERE "
        where += prefix + "parent_task_id = ?"
        params.append(parent_task_id)
    with _db() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM requests{where}", params).fetchone()[0]
        rows = [dict(r) for r in conn.execute(
            f"SELECT * FROM requests{where} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )]
    rows = [_task_row(row) for row in rows]
    return {
        "ok": True,
        "items": rows,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(rows) < total,
    }


@app.get("/api/admin/tasks/{task_id}")
def admin_task(
    task_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_admin(request, authorization, x_admin_token, token)
    with _db() as conn:
        row = conn.execute("SELECT * FROM requests WHERE task_id = ?", (task_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "task not found")
    return {"ok": True, "task": _task_row(row)}


@app.get("/api/admin/files")
def admin_files(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(60, ge=1, le=500),
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_admin(request, authorization, x_admin_token, token)
    items, total = _static_files(offset=offset, limit=limit, include_task_id=True)
    return {"ok": True, "items": items, "total": total, "offset": offset, "limit": limit, "has_more": offset + len(items) < total}


@app.post("/api/admin/cleanup")
def admin_cleanup(
    request: Request,
    ttl_hours: int | None = Query(default=None, ge=1, le=24 * 365),
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_admin(request, authorization, x_admin_token, token)
    if ttl_hours is None:
        ttl_hours = _get_int_setting("image_ttl_hours", IMAGE_TTL_HOURS, 1, 24 * 365)
    return {"ok": True, **_cleanup_static(ttl_hours)}


class AdminSettingsRequest(BaseModel):
    max_concurrent_tasks: Optional[int] = Field(default=None, ge=1, le=128)
    max_retries: Optional[int] = Field(default=None, ge=0, le=10)
    retry_backoff_seconds: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    image_ttl_hours: Optional[int] = Field(default=None, ge=1, le=24 * 365)


@app.get("/api/admin/settings")
def admin_get_settings(
    request: Request,
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_admin(request, authorization, x_admin_token, token)
    with _INFLIGHT_LOCK:
        inflight = _INFLIGHT_TASKS
    return {
        "ok": True,
        "settings": {
            "max_concurrent_tasks": _get_int_setting("max_concurrent_tasks", DEFAULT_MAX_CONCURRENT_TASKS, 1, 128),
            "max_retries": _get_int_setting("max_retries", MAX_RETRIES, 0, 10),
            "retry_backoff_seconds": _get_float_setting("retry_backoff_seconds", RETRY_BACKOFF_SECONDS, 0.0, 10.0),
            "image_ttl_hours": _get_int_setting("image_ttl_hours", IMAGE_TTL_HOURS, 1, 24 * 365),
            "inflight_tasks": inflight,
        },
    }


@app.post("/api/admin/settings")
def admin_set_settings(
    payload: AdminSettingsRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_admin(request, authorization, x_admin_token, token)
    values = {k: v for k, v in payload.model_dump().items() if v is not None}
    return {"ok": True, "saved": _set_settings(values)}


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return HTMLResponse(content=ADMIN_HTML, headers={"Cache-Control": "no-store, max-age=0"})


ADMIN_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FaceBlur Admin</title>
  <style>
    :root {
      --bg: #f4f2ec;
      --panel: #fffdf7;
      --ink: #171717;
      --muted: #6f6b62;
      --line: #d8d2c4;
      --accent: #1f7a5a;
      --warn: #a84526;
      --shadow: 0 12px 30px rgba(32, 29, 22, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-serif, Georgia, "Times New Roman", "Microsoft YaHei", serif;
    }
    button, input, select { font: inherit; }
    .shell { max-width: 1920px; margin: 0 auto; padding: 28px 24px 48px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: flex-end; margin-bottom: 22px; }
    h1 { margin: 0; font-size: 34px; line-height: 1; letter-spacing: 0; }
    .sub { margin-top: 8px; color: var(--muted); font-size: 14px; }
    .auth { display: flex; gap: 8px; align-items: center; }
    .auth input { width: 260px; padding: 10px 12px; border: 1px solid var(--line); background: var(--panel); border-radius: 6px; }
    .btn { border: 1px solid var(--ink); background: var(--ink); color: white; padding: 10px 14px; border-radius: 6px; cursor: pointer; }
    .btn.secondary { background: transparent; color: var(--ink); border-color: var(--line); }
    .btn.danger { background: var(--warn); border-color: var(--warn); }
    .btn:disabled { cursor: not-allowed; opacity: .45; }
    .tabs { display: flex; gap: 4px; margin-bottom: 18px; border-bottom: 1px solid var(--line); }
    .tab { border: 0; border-bottom: 3px solid transparent; background: transparent; color: var(--muted); padding: 10px 16px; cursor: pointer; }
    .tab.active { border-bottom-color: var(--accent); color: var(--ink); font-weight: 700; }
    .tab-view[hidden] { display: none; }
    .grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: var(--shadow); min-height: 104px; }
    .label { color: var(--muted); font-size: 13px; }
    .metric { margin-top: 8px; font-size: 30px; font-weight: 700; }
    .split { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(360px, 1fr); gap: 18px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; box-shadow: var(--shadow); }
    .panel-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 14px 16px; border-bottom: 1px solid var(--line); }
    .settings { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; padding: 14px 16px; align-items: end; }
    .field label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .field input { width: 100%; padding: 9px 10px; border: 1px solid var(--line); border-radius: 6px; background: white; }
    h2 { margin: 0; font-size: 18px; }
    table { width: 100%; border-collapse: collapse; font-family: "Microsoft YaHei", Arial, sans-serif; font-size: 13px; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #ebe5d8; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 600; background: #faf7ef; }
    .pill { display: inline-flex; padding: 3px 8px; border-radius: 999px; background: #e8f1eb; color: var(--accent); font-size: 12px; }
    .pill.err { background: #f7e6df; color: var(--warn); }
    .files { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; padding: 14px; }
    .gallery-files { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .file { border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background: white; }
    .file img { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: cover; background: #eee; }
    .file div { padding: 8px; color: var(--muted); font-family: "Microsoft YaHei", Arial, sans-serif; font-size: 12px; word-break: break-all; }
    .status { margin: 12px 0; color: var(--muted); font-family: "Microsoft YaHei", Arial, sans-serif; }
    .gallery-tools, .request-tools, .task-search, .pager { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .gallery-tools select, .request-tools select, .task-search input { padding: 9px 10px; border: 1px solid var(--line); border-radius: 6px; background: white; }
    .task-search input { width: min(360px, 100%); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .pager { justify-content: center; padding: 14px 16px 18px; border-top: 1px solid var(--line); }
    .pager-info { min-width: 180px; color: var(--muted); text-align: center; font-family: "Microsoft YaHei", Arial, sans-serif; font-size: 13px; }
    .task-id { max-width: 280px; overflow: hidden; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
    .link-btn { border: 0; background: transparent; color: var(--accent); padding: 4px 0; cursor: pointer; white-space: nowrap; }
    .task-body { padding: 18px; }
    .task-heading { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 16px; }
    .task-heading h2 { margin-bottom: 7px; }
    .task-heading code { color: var(--muted); word-break: break-all; }
    .task-facts { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1px; margin-bottom: 18px; border: 1px solid var(--line); background: var(--line); }
    .fact { min-height: 78px; padding: 12px; background: white; }
    .fact strong { display: block; margin-top: 7px; word-break: break-word; }
    .task-images { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
    .image-preview { border: 1px solid var(--line); background: white; }
    .image-preview h3, .json-block h3 { margin: 0; padding: 11px 13px; border-bottom: 1px solid var(--line); font: 600 13px "Microsoft YaHei", Arial, sans-serif; }
    .image-preview img { display: block; width: 100%; height: 320px; object-fit: contain; background: #efede7; }
    .image-empty { display: grid; height: 320px; place-items: center; color: var(--muted); }
    .task-json { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .json-block { min-width: 0; border: 1px solid var(--line); background: white; }
    .json-block pre { min-height: 220px; max-height: 520px; margin: 0; padding: 13px; overflow: auto; background: #f7f4ec; white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.55; }
    @media (max-width: 860px) { .grid, .split, .settings, .task-facts, .task-images, .task-json { grid-template-columns: 1fr; } header, .task-heading { align-items: stretch; flex-direction: column; } .auth input, .task-search input { width: 100%; } .files, .gallery-files { grid-template-columns: 1fr; } table { min-width: 760px; } .panel { overflow-x: auto; } }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <a href="/admin" style="text-decoration:none;color:inherit"><h1>FaceBlur Admin</h1></a>
        <div class="sub">Live request log, static image storage, cleanup control</div>
      </div>
      <div class="auth">
        <input id="token" type="password" placeholder="Admin token" autocomplete="off" />
        <button class="btn" onclick="saveToken()">保存</button>
        <button class="btn secondary" onclick="loadAll()">刷新</button>
      </div>
    </header>

    <nav class="tabs" aria-label="管理页面标签">
      <button class="tab active" data-tab="overview" onclick="showTab('overview')">概览</button>
      <button class="tab" data-tab="gallery" onclick="showGalleryTab()">图片库</button>
      <button class="tab" id="task-tab" data-tab="task" onclick="showTab('task')" hidden>任务详情</button>
    </nav>
    <div class="task-search">
      <input id="task-search" type="search" placeholder="输入任务 ID 或父任务ID 定位" aria-label="任务 ID" />
      <button class="btn secondary" onclick="findTask()">查询任务</button>
    </div>

    <div class="status" id="status">正在加载...</div>
    <div class="tab-view" data-tab="overview">
    <section class="grid">
      <div class="card"><div class="label">总请求</div><div class="metric" id="m-total">-</div></div>
      <div class="card"><div class="label">已打码</div><div class="metric" id="m-blocked">-</div></div>
      <div class="card"><div class="label">成功率</div><div class="metric" id="m-rate">-</div></div>
      <div class="card"><div class="label">24h 成功率</div><div class="metric" id="m-rate-24h">-</div></div>
      <div class="card"><div class="label">静态图片</div><div class="metric" id="m-files">-</div></div>
    </section>

    <section class="panel" style="margin-bottom:14px">
      <div class="panel-head"><h2>运行设置</h2><button class="btn secondary" onclick="saveSettings()">保存设置</button></div>
      <div class="settings">
        <div class="field"><label>并行任务上限</label><input id="s-concurrency" type="number" min="1" max="32" /></div>
        <div class="field"><label>失败重试次数</label><input id="s-retries" type="number" min="0" max="10" /></div>
        <div class="field"><label>重试退避秒</label><input id="s-backoff" type="number" min="0" max="10" step="0.1" /></div>
        <div class="field"><label>图片保留小时</label><input id="s-ttl" type="number" min="1" max="8760" /></div>
      </div>
    </section>

    <section class="split">
      <div class="panel">
        <div class="panel-head">
          <h2>请求记录</h2>
          <div class="request-tools">
            <label for="request-page-size">每页显示</label>
            <select id="request-page-size" onchange="changeRequestPageSize()">
              <option value="10">10</option>
              <option value="20" selected>20</option>
              <option value="50">50</option>
              <option value="100">100</option>
              <option value="200">200</option>
            </select>
            <label for="status-filter">状态</label>
            <select id="status-filter" onchange="loadRequests()">
              <option value="">全部</option>
              <option value="ok">ok</option>
              <option value="rejected">rejected</option>
              <option value="download_error">download_error</option>
              <option value="bad_request">bad_request</option>
              <option value="process_error">process_error</option>
              <option value="validation_error">validation_error</option>
            </select>
            <button class="btn secondary" onclick="loadRequests()">刷新</button>
          </div>
        </div>
        <table>
          <thead><tr><th>时间</th><th>任务 ID</th><th>父任务ID</th><th>状态</th><th>模式</th><th>来源</th><th>耗时</th><th>操作</th></tr></thead>
          <tbody id="requests"></tbody>
        </table>
        <div class="pager">
          <button class="btn secondary" id="request-prev" onclick="changeRequestPage(-1)">上一页</button>
          <span class="pager-info" id="request-page-info">第 1 / 1 页</span>
          <input type="number" id="request-jump-page" min="1" placeholder="页" style="width:48px;text-align:center" onkeydown="if(event.key==='Enter')jumpRequestPage()">
          <button class="btn secondary" onclick="jumpRequestPage()">跳转</button>
          <button class="btn secondary" id="request-next" onclick="changeRequestPage(1)">下一页</button>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h2>图片库</h2>
          <div><button class="btn secondary" onclick="showGalleryTab()">加载更多</button> <button class="btn danger" onclick="cleanup()">清理过期</button></div>
        </div>
        <div class="files" id="files"></div>
      </div>
    </section>
    </div>

    <div class="tab-view" data-tab="gallery" hidden>
      <section class="panel">
        <div class="panel-head">
          <h2>图片库</h2>
          <div class="gallery-tools">
            <label for="gallery-page-size">每页显示</label>
            <select id="gallery-page-size" onchange="changeGalleryPageSize()">
              <option value="10" selected>10</option>
              <option value="20">20</option>
              <option value="50">50</option>
              <option value="100">100</option>
              <option value="200">200</option>
            </select>
            <button class="btn secondary" onclick="loadGalleryPage()">刷新</button>
            <button class="btn danger" onclick="cleanup()">清理过期</button>
          </div>
        </div>
        <div class="files gallery-files" id="gallery-files"></div>
        <div class="pager">
          <button class="btn secondary" id="gallery-prev" onclick="changeGalleryPage(-1)">上一页</button>
          <span class="pager-info" id="gallery-page-info">第 1 / 1 页</span>
          <input type="number" id="gallery-jump-page" min="1" placeholder="页" style="width:48px;text-align:center" onkeydown="if(event.key==='Enter')jumpGalleryPage()">
          <button class="btn secondary" onclick="jumpGalleryPage()">跳转</button>
          <button class="btn secondary" id="gallery-next" onclick="changeGalleryPage(1)">下一页</button>
        </div>
      </section>
    </div>

    <div class="tab-view" data-tab="task" hidden>
      <section class="panel">
        <div class="task-body" id="task-detail">
          <div class="status">请从请求记录选择任务，或输入任务 ID 查询。</div>
        </div>
      </section>
    </div>
  </main>
  <script>
    const tokenEl = document.getElementById('token');
    let activeTab = 'overview';
    let requestPage = 1;
    let requestPageSize = 20;
    let requestPageCount = 1;
    let galleryPage = 1;
    let galleryPageSize = 10;
    let galleryPageCount = 1;
    tokenEl.value = localStorage.getItem('faceblur_admin_token') || '';
    function persistTokenAndReload(){
      const value = tokenEl.value.trim();
      if(value) localStorage.setItem('faceblur_admin_token', value);
      else localStorage.removeItem('faceblur_admin_token');
      loadAll();
    }
    function saveToken(){ persistTokenAndReload(); }
    function headers(){ const t = tokenEl.value.trim(); return t ? {'X-Admin-Token': t} : {}; }
    function setStatus(text){ document.getElementById('status').textContent = text; }
    function fmtBytes(n){ if(!n) return '0 B'; const u=['B','KB','MB','GB']; let i=0; while(n>=1024&&i<u.length-1){n/=1024;i++;} return `${n.toFixed(i?1:0)} ${u[i]}`; }
    function escapeHtml(value){
      return String(value ?? '').replace(/[&<>"']/g, char => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
      })[char]);
    }
    function fmtBeijingTime(value){
      if(!value) return '-';
      const date = new Date(value);
      if(Number.isNaN(date.getTime())) return value;
      return new Intl.DateTimeFormat('zh-CN', {
        timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
      }).format(date).replaceAll('/', '-');
    }
    async function api(path, opts={}){
      if(!tokenEl.value.trim()){
        throw new Error('请先在右上角输入 Admin token');
      }
      const r = await fetch(path, {...opts, headers: {...headers(), ...(opts.headers||{})}});
      if(!r.ok){
        const text = await r.text();
        if(r.status === 401){
          throw new Error(`token 无效或缺失（401 unauthorized）— 请检查右上角的 Admin token 是否填对。服务器返回：${text}`);
        }
        throw new Error(`${r.status} ${text}`);
      }
      return r.json();
    }
    async function loadSummary(){
      const d = await api('/api/admin/summary');
      document.getElementById('m-total').textContent = d.requests.total;
      document.getElementById('m-blocked').textContent = d.requests.blocked;
      document.getElementById('m-rate').textContent = `${d.requests.success_rate}%`;
      document.getElementById('m-rate-24h').textContent = `${d.requests.success_rate_24h}%`;
      document.getElementById('m-files').textContent = `${d.storage.file_count}`;
      document.getElementById('s-concurrency').value = d.service.max_concurrent_tasks;
      document.getElementById('s-retries').value = d.service.max_retries;
      document.getElementById('s-backoff').value = d.service.retry_backoff_seconds;
      document.getElementById('s-ttl').value = d.service.image_ttl_hours_effective || d.service.image_ttl_hours;
      setStatus(`并行 ${d.service.inflight_tasks}/${d.service.max_concurrent_tasks} · 24h ${d.requests.last_24h_ok}/${d.requests.last_24h} 成功 · 重试 ${d.requests.retried} 次 · 存储 ${fmtBytes(d.storage.bytes)} · PID ${d.service.pid}`);
    }
    function requestRecord(x){
      return {
        id: x.id,
        time_beijing: fmtBeijingTime(x.created_at),
        time_utc: x.created_at,
        request: x.request || '历史记录未保存完整请求参数',
        result: {status:x.status, mode:x.mode, blocked:Boolean(x.blocked), face_count:x.face_count, error:x.error},
        timing: {elapsed_ms:x.elapsed_ms, process_ms:x.process_ms, attempts:x.attempts, retried:Boolean(x.retried)},
        transfer: {input_bytes:x.input_bytes, output_bytes:x.output_bytes, image_url:x.image_url, output_url:x.output_url, output_file:x.output_file},
        client: {ip:x.client_ip, user_agent:x.user_agent},
      };
    }
    function jsonView(value){ return escapeHtml(JSON.stringify(value, null, 2)); }
    function showImgSize(img){
      const el = document.getElementById(img.id + "-size");
      const dims = img.naturalWidth + " × " + img.naturalHeight + " px";
      const bytes = parseInt(img.dataset.size || 0);
      el.textContent = dims + (bytes ? " · " + fmtBytes(bytes) : "");
    }
    function imageView(id, label, url, sizeBytes){
      const meta = id + "-size";
      const sizeText = sizeBytes ? ` \xb7 ${fmtBytes(sizeBytes)}` : "";
      return `<div class="image-preview"><h3><span>${label}</span><span class="img-meta" id="${meta}">${sizeText}</span></h3>${url
        ? `<img id="${id}" src="${escapeHtml(url)}" alt="${label}" loading="lazy" onload="showImgSize(this)" data-size="${sizeBytes || 0}" />`
        : `<div class="image-empty" id="${id}">无可预览图片</div>`}</div>`;
    }
    async function loadRequests(){
      let url = '/api/admin/requests?offset=' + ((requestPage - 1) * requestPageSize) + '&limit=' + requestPageSize;
      const q = document.getElementById('task-search').value.trim();
      if(q && q.length !== 32) url += '&parent_task_id=' + encodeURIComponent(q);
      const st = document.getElementById('status-filter').value;
      if(st) url += '&status=' + encodeURIComponent(st);
      const d = await api(url);
      requestPageCount = Math.max(1, Math.ceil(d.total / requestPageSize));
      if(requestPage > requestPageCount){ requestPage = requestPageCount; return loadRequests(); }
      document.getElementById('requests').innerHTML = d.items.map(x => `
        <tr>
          <td>${fmtBeijingTime(x.created_at)}</td>
          <td><div class="task-id" title="${escapeHtml(x.task_id)}">${escapeHtml(x.task_id || '-')}</div></td>
          <td>${escapeHtml(x.parent_task_id || "-")}</td>
          <td><span class="pill ${x.status === 'ok' ? '' : 'err'}">${x.status}</span></td>
          <td>${escapeHtml(x.mode || '-')}</td>
          <td>${escapeHtml(x.client_ip || '-')}</td>
          <td>${Math.round(x.elapsed_ms)}ms</td>
          <td><button class="link-btn" onclick="showTaskDetail('${escapeHtml(x.task_id)}')">查看任务</button></td>
        </tr>`).join('');
      document.getElementById('request-page-info').textContent = `第 ${requestPage} / ${requestPageCount} 页，共 ${d.total} 条`;
      document.getElementById('request-prev').disabled = requestPage <= 1;
      document.getElementById('request-next').disabled = requestPage >= requestPageCount;
    }
    async function changeRequestPage(delta){
      const nextPage = Math.min(requestPageCount, Math.max(1, requestPage + delta));
      if(nextPage === requestPage) return;
      requestPage = nextPage;
      await loadRequests();
    }
    async function jumpRequestPage(){
      const p = parseInt(document.getElementById("request-jump-page").value);
      if(p >= 1 && p <= requestPageCount){ requestPage = p; await loadRequests(); }
    }
    async function jumpGalleryPage(){
      const p = parseInt(document.getElementById("gallery-jump-page").value);
      if(p >= 1 && p <= galleryPageCount){ galleryPage = p; await loadGalleryPage(); window.scrollTo({top: 0, behavior: "smooth"}); }
    }
    async function changeRequestPageSize(){
      requestPageSize = Number(document.getElementById('request-page-size').value);
      requestPage = 1;
      await loadRequests();
    }
    async function loadFiles(){
      const d = await api('/api/admin/files?offset=0&limit=10');
      document.getElementById('files').innerHTML = d.items.map(x => {
        const tid = x.task_id || '';
        const onclick = tid ? `onclick="showTaskDetail('${escapeHtml(tid)}')"` : '';
        const style = tid ? 'style="cursor:pointer"' : '';
        const title = tid ? 'title="点击查看任务详情"' : '';
        return `<div class="file" ${onclick} ${style} ${title}><img src="${x.url}" loading="lazy" /><div>${x.name}<br>${fmtBytes(x.size)}</div></div>`;
      }).join('') || '<div class="status">暂无图片</div>';
    }
    async function showTaskDetail(taskId){
      if(!taskId) return;
      try {
        const d = await api(`/api/admin/tasks/${encodeURIComponent(taskId)}`);
        const x = d.task;
        const record = requestRecord(x);
        const inputUrl = x.image_url || x.request?.body?.image_url || '';
        const outputUrl = x.output_url || '';
        const sameImage = inputUrl && outputUrl && inputUrl === outputUrl;
        document.getElementById('task-tab').hidden = false;
        document.getElementById('task-search').value = x.task_id;
        document.getElementById('task-detail').innerHTML = `
          <div class="task-heading">
            <div><h2>任务详情</h2><code>${escapeHtml(x.task_id)}</code></div>
            <button class="btn secondary" onclick="showTab('overview');window.scrollTo({top:0,behavior:'smooth'})">返回首页</button>
            <button class="btn secondary" onclick="copyTaskId()">复制任务 ID</button>
          </div>
          <div class="task-facts">
            <div class="fact"><span class="label">状态</span><strong><span class="pill ${x.status === 'ok' ? '' : 'err'}">${escapeHtml(x.status)}</span></strong></div>
            <div class="fact"><span class="label">北京时间</span><strong>${fmtBeijingTime(x.created_at)}</strong></div>
            <div class="fact"><span class="label">模式</span><strong>${escapeHtml(x.mode || '-')}</strong></div>
            <div class="fact"><span class="label">人脸数量</span><strong>${Number(x.face_count || 0)}</strong></div>
            <div class="fact"><span class="label">总耗时</span><strong>${Math.round(x.elapsed_ms || 0)} ms</strong></div>
          </div>
          ${x.error ? `<div class="status" style="color:var(--warn)">失败原因：${escapeHtml(x.error)}</div>` : ''}
          <div class="task-images">
            ${imageView('task-input-image', '输入图片', inputUrl, x.input_bytes)}
            ${imageView('task-output-image', sameImage ? '结果图片（未检测到人脸，返回原图）' : '生成结果', outputUrl, x.output_bytes)}
          </div>
          <div class="task-json">
            <div class="json-block"><h3>完整请求信息</h3><pre id="task-request-json">${jsonView(x.request || record.request)}</pre></div>
            <div class="json-block"><h3>完整响应与执行信息</h3><pre id="task-response-json">${jsonView({response:x.response, result:record.result, timing:record.timing, transfer:record.transfer, client:record.client})}</pre></div>
          </div>`;
        showTab('task');
        history.replaceState(null, '', `#task=${encodeURIComponent(taskId)}`);
        window.scrollTo({top: 0, behavior: 'smooth'});
      } catch(e) { setStatus(`任务查询失败: ${e.message}`); }
    }
    function findTask(){
      const q = document.getElementById('task-search').value.trim();
      if(!q) return;
      if(q.length === 32) showTaskDetail(q);
      else { requestPage = 1; loadRequests(); }
    }
    async function copyTaskId(){
      const taskId = document.getElementById('task-search').value.trim();
      if(!taskId) return;
      await navigator.clipboard.writeText(taskId);
      setStatus('任务 ID 已复制');
    }
    function showTab(name){
      activeTab = name;
      document.querySelectorAll('.tab-view').forEach(el => { el.hidden = el.dataset.tab !== name; });
      document.querySelectorAll('.tab').forEach(el => el.classList.toggle('active', el.dataset.tab === name));
    }
    async function showGalleryTab(){
      showTab('gallery');
      await loadGalleryPage();
    }
    async function loadGalleryPage(){
      const d = await api(`/api/admin/files?offset=${(galleryPage - 1) * galleryPageSize}&limit=${galleryPageSize}`);
      galleryPageCount = Math.max(1, Math.ceil(d.total / galleryPageSize));
      if(galleryPage > galleryPageCount){ galleryPage = galleryPageCount; return loadGalleryPage(); }
      document.getElementById('gallery-files').innerHTML = d.items.map(x => {
        const tid = x.task_id || '';
        const onclick = tid ? `onclick="showTaskDetail('${escapeHtml(tid)}')"` : '';
        const style = tid ? 'style="cursor:pointer"' : '';
        const title = tid ? 'title="点击查看任务详情"' : '';
        return `<div class="file" ${onclick} ${style} ${title}><img src="${x.url}" loading="lazy" /><div>${x.name}<br>${fmtBytes(x.size)}</div></div>`;
      }).join('') || '<div class="status">暂无图片</div>';
      document.getElementById('gallery-page-info').textContent = `第 ${galleryPage} / ${galleryPageCount} 页，共 ${d.total} 张`;
      document.getElementById('gallery-prev').disabled = galleryPage <= 1;
      document.getElementById('gallery-next').disabled = galleryPage >= galleryPageCount;
      setStatus(`图片库第 ${galleryPage} 页，每页 ${galleryPageSize} 张`);
    }
    async function changeGalleryPage(delta){
      const nextPage = Math.min(galleryPageCount, Math.max(1, galleryPage + delta));
      if(nextPage === galleryPage) return;
      galleryPage = nextPage;
      await loadGalleryPage();
      window.scrollTo({top: 0, behavior: 'smooth'});
    }
    async function changeGalleryPageSize(){
      galleryPageSize = Number(document.getElementById('gallery-page-size').value);
      galleryPage = 1;
      await loadGalleryPage();
    }
    async function saveSettings(){
      const body = {
        max_concurrent_tasks: Number(document.getElementById('s-concurrency').value),
        max_retries: Number(document.getElementById('s-retries').value),
        retry_backoff_seconds: Number(document.getElementById('s-backoff').value),
        image_ttl_hours: Number(document.getElementById('s-ttl').value),
      };
      await api('/api/admin/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      await loadSummary();
      setStatus('设置已保存');
    }
    async function cleanup(){
      if(!confirm('按当前 TTL 清理过期静态图片？')) return;
      const d = await api('/api/admin/cleanup', {method:'POST'});
      setStatus(`已删除 ${d.deleted_files} 个文件，释放 ${fmtBytes(d.freed_bytes)}`);
      await loadAll();
    }
    async function loadAll(){
      try {
        await Promise.all([loadSummary(), loadRequests(), loadFiles()]);
        if(activeTab === 'gallery') await loadGalleryPage();
      }
      catch(e){ setStatus(`加载失败: ${e.message}`); }
    }
    document.getElementById('task-search').addEventListener('keydown', event => {
      if(event.key === 'Enter') findTask();
    });
    tokenEl.addEventListener('input', () => { localStorage.setItem('faceblur_admin_token', tokenEl.value.trim()); });
    tokenEl.addEventListener('change', () => { persistTokenAndReload(); });
    tokenEl.addEventListener('keydown', event => { if(event.key === 'Enter'){ event.preventDefault(); persistTokenAndReload(); } });
    loadAll().then(() => {
      const match = location.hash.match(/^#task=(.+)$/);
      if(match) showTaskDetail(decodeURIComponent(match[1]));
    });
    setInterval(() => { if(tokenEl.value.trim()) loadAll(); }, 30000);
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
