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
from pydantic import BaseModel, Field, HttpUrl, model_validator
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
MAX_IMAGE_BYTES = int(os.environ.get("FACE_BLUR_MAX_BYTES", str(50 * 1024 * 1024)))  # 50 MB
IMAGE_TTL_HOURS = int(os.environ.get("FACE_BLUR_IMAGE_TTL_HOURS", "72"))
DB_CACHE_TTL_HOURS = int(os.environ.get("FACE_BLUR_DB_CACHE_TTL_HOURS", str(IMAGE_TTL_HOURS)))
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("FACE_BLUR_CLEANUP_INTERVAL_SECONDS", "3600"))
MAX_RETRIES = int(os.environ.get("FACE_BLUR_MAX_RETRIES", "2"))
RETRY_BACKOFF_SECONDS = float(os.environ.get("FACE_BLUR_RETRY_BACKOFF_SECONDS", "0.6"))
DEFAULT_MAX_CONCURRENT_TASKS = int(os.environ.get("FACE_BLUR_MAX_CONCURRENT_TASKS", "4"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("faceblur-api")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="face_blur API", version="1.1.0",
              description="人脸打码服务 (pixelate/gaussian/solid/landmark/landmark_whole_face)")

# 静态文件服务(返回打码后的图)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_db() -> None:
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inflight_counter (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                n INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute("INSERT OR IGNORE INTO inflight_counter (id, n) VALUES (1, 0)")
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
            "CREATE INDEX IF NOT EXISTS idx_requests_parent_output ON requests(parent_task_id, output_file)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_parent_id ON requests(parent_task_id, id DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_output_file ON requests(output_file)"
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
            "cache_epoch": "0",
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
        # Migration: add face_count column (safe to run repeatedly)
        try:
            conn.execute("ALTER TABLE blur_cache ADD COLUMN face_count INTEGER DEFAULT 0")
        except Exception:
            pass


def _get_setting(key: str, default: str) -> str:
    try:
        with _db() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default
    except Exception as e:  # noqa: BLE001
        log.warning("settings read failed: %s", e)
        return default


def _get_blur_default(key: str, default):
    """读取全局打码默认参数, 类型不匹配则回退默认值"""
    try:
        with _db() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if row:
            v = json.loads(row[0])
            return v if type(v) == type(default) else default
    except Exception:
        pass
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
        "score_threshold": (0.1, 0.99, float),
        "expand_ratio": (0.0, 1.0, float),
        "min_face_skip": (0, 500, int),
        "dot_radius": (1, 20, int),
        "face_grid_step": (4, 60, int),
        "grid_n": (1, 11, int),
        "face_profiles": (None, None, list),
    }
    saved = {}
    with _db() as conn:
        for key, raw in values.items():
            if key not in allowed:
                continue
            lo, hi, caster = allowed[key]
            if key == "face_profiles":
                val = raw
                text = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
            else:
                val = caster(raw)
                val = max(lo, min(hi, val))
                text = str(val)
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, text, _utc_now()),
            )
            saved[key] = val
    return saved


def _bump_cache_epoch() -> str:
    epoch = uuid.uuid4().hex
    with _db() as conn:
        conn.execute(
            """INSERT INTO settings (key, value, updated_at) VALUES ('cache_epoch', ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (epoch, _utc_now()),
        )
    return epoch


def _inflight_try_acquire(limit: int) -> bool:
    """原子操作: n < limit 时 +1, 否则返回 False (被拒绝)."""
    for _ in range(3):
        try:
            with _db() as conn:
                conn.execute(
                    "UPDATE inflight_counter SET n = n + 1 WHERE id = 1 AND n < ?",
                    (limit,),
                )
                return conn.total_changes > 0
        except Exception:
            time.sleep(0.02)
    return False

def _inflight_release() -> None:
    """释放并发槽位 (n - 1)."""
    for _ in range(3):
        try:
            with _db() as conn:
                conn.execute(
                    "UPDATE inflight_counter SET n = MAX(0, n - 1) WHERE id = 1"
                )
            return
        except Exception:
            time.sleep(0.02)

def _inflight_read() -> int:
    try:
        with _db() as conn:
            row = conn.execute("SELECT n FROM inflight_counter WHERE id = 1").fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


@contextmanager
def _task_slot():
    limit = _get_int_setting("max_concurrent_tasks", DEFAULT_MAX_CONCURRENT_TASKS, 1, 128)
    deadline = time.perf_counter() + 30
    attempt = 0
    while True:
        if _inflight_try_acquire(limit):
            break
        attempt += 1
        if time.perf_counter() >= deadline:
            raise HTTPException(
                429,
                f"too many concurrent tasks (limit {limit}), gave up after {attempt} retries",
            )
        time.sleep(min(0.5 * attempt, 3))
    try:
        yield limit
    finally:
        _inflight_release()


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
                  include_task_id: bool = False,
                  parent_task_id: str | None = None) -> tuple[list[dict], int]:
    parent_task_map = {}
    if parent_task_id:
        paths = []
        try:
            with _db() as conn:
                rows = conn.execute(
                    """SELECT output_file, output_url, task_id FROM requests
                       WHERE parent_task_id=? AND (output_file IS NOT NULL OR output_url IS NOT NULL)
                       ORDER BY id DESC""",
                    (parent_task_id,),
                ).fetchall()
                allowed = []
                seen = set()
                for row in rows:
                    name = Path(row["output_file"]).name if row["output_file"] else _static_name_from_url(row["output_url"])
                    if name.endswith(".jpg") and name not in seen:
                        allowed.append(name)
                        seen.add(name)
                    if name.endswith(".jpg") and name not in parent_task_map:
                        parent_task_map[name] = row["task_id"]
                candidates = [STATIC_DIR / name for name in allowed]
                paths = [path for path in candidates if path.is_file()]
        except Exception as e:
            log.warning("parent gallery lookup failed: %s", e)
    else:
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
            if parent_task_id:
                task_map = {path.name: parent_task_map.get(path.name, "") for path in paths}
            else:
                file_names = [p.name for p in paths]
                with _db() as conn:
                    placeholders = ",".join("?" for _ in file_names)
                    rows = conn.execute(
                        f"SELECT output_file, task_id FROM requests WHERE output_file IN ({placeholders}) ORDER BY id DESC",
                        file_names,
                    ).fetchall()
                    for row in rows:
                        task_map.setdefault(row["output_file"], row["task_id"])
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


def _static_name_from_url(url: str | None) -> str:
    """从本服务静态图片 URL 兼容推导文件名，覆盖旧缓存记录。"""
    if not url:
        return ""
    try:
        path = _urlparse_mod.urlparse(str(url)).path
        if "/static/" not in path:
            return ""
        name = Path(path).name
        return name if name.endswith(".jpg") else ""
    except Exception:
        return ""


_STORAGE_STATS_LOCK = threading.Lock()
_STORAGE_STATS_CACHE = {"expires": 0.0, "file_count": 0, "bytes": 0}


def _storage_stats() -> dict:
    now = time.monotonic()
    with _STORAGE_STATS_LOCK:
        if _STORAGE_STATS_CACHE["expires"] > now:
            return {"file_count": _STORAGE_STATS_CACHE["file_count"], "bytes": _STORAGE_STATS_CACHE["bytes"]}
        file_count = 0
        total_bytes = 0
        for path in STATIC_DIR.glob("*.jpg"):
            try:
                total_bytes += path.stat().st_size
                file_count += 1
            except FileNotFoundError:
                continue
        _STORAGE_STATS_CACHE.update(expires=now + 15.0, file_count=file_count, bytes=total_bytes)
        return {"file_count": file_count, "bytes": total_bytes}


def _invalidate_storage_stats() -> None:
    with _STORAGE_STATS_LOCK:
        _STORAGE_STATS_CACHE["expires"] = 0.0


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
        if deleted:
            _invalidate_storage_stats()
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
    image_url: HttpUrl | None = None
    mode: str = Field(
        "gaussian",
        pattern=r"^(pixelate|gaussian|solid|landmark|landmark_whole_face)$",
    )
    modes: list[str] = Field(default_factory=list, max_length=5)
    face_profiles: list[dict] = Field(default_factory=list, max_length=3)
    # 默认值已对齐 face_blur.py / v14r3 推荐参数 (2026-08-05)
    score_threshold: float = Field(0.52, ge=0.1, le=0.99)
    expand_ratio: float = Field(0.30, ge=0.0, le=1.0)
    dot_radius: int = Field(3, ge=1, le=20)
    spacing: int = Field(14, ge=4, le=60)
    # landmark_whole_face 模式专用参数
    face_grid_step: int = Field(14, ge=4, le=60)
    grid_n: int = Field(5, ge=3, le=11, description="关键点附近矩阵大小 (建议 3-7)")
    min_face_skip: int = Field(40, ge=0, le=500, description="极小人脸跳过阈值")
    parent_task_id: Optional[str] = Field(None, max_length=200, description="上游任务批次标记")
    image_base64: Optional[str] = Field(None, description="Base64 image for lab test")
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
    parent_task_id = body.get("parent_task_id") if isinstance(body, dict) else None
    response = {"task_id": task_id, "status": "validation_error", "error": error, "errors": exc.errors()}
    _insert_request({
        "task_id": task_id,
        "status": "validation_error",
        "mode": body.get("mode") if isinstance(body, dict) else None,
        "image_url": body.get("image_url") if isinstance(body, dict) else None,
        "parent_task_id": parent_task_id,
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
        # 图片源应由服务直接访问，避免本地/运行环境的 HTTP(S) 代理导致
        # 第三方 CDN TLS 握手被代理的短超时中断。
        sess.trust_env = False
        _adapter = _rq.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=10)
        sess.mount("https://", _adapter)
        sess.mount("http://", _adapter)
        _DL_LOCAL.session = sess
    return sess

def _download(url: str, max_bytes: int = MAX_IMAGE_BYTES,
              timeout: int = DOWNLOAD_TIMEOUT) -> bytes:
    """下载 URL, 大图分块并发 (4 线程), 小图单连接."""
    _assert_public_url(url)
    sess = _dl_session()
    ua = {"User-Agent": "faceblur-api/1.0"}

    # 探测: HEAD 拿文件大小, 不支持 Range 就走单线程
    total_size = 0
    accept_ranges = False
    try:
        with sess.head(url, timeout=(5, 10), headers=ua) as hdr:
            hdr.raise_for_status()
            total_size = int(hdr.headers.get("Content-Length", 0))
            accept_ranges = hdr.headers.get("Accept-Ranges", "") == "bytes"
    except Exception:
        pass

    # 小文件 / 不支持 Range / HEAD 失败 → 单线程
    CHUNK_THRESHOLD = 1_000_000  # 1 MB
    if total_size < CHUNK_THRESHOLD or not accept_ranges:
        with sess.get(url, timeout=(5, timeout), stream=True, headers=ua) as resp:
            resp.raise_for_status()
            data = b""
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    break
                data += chunk
                if len(data) > max_bytes:
                    raise HTTPException(413, f"image too large (> {max_bytes} bytes)")
        return data

    if total_size > max_bytes:
        raise HTTPException(413, f"image too large ({total_size} > {max_bytes} bytes)")

    # 大文件: 4 线程分块并发下载
    NUM = 8
    chunk_size = (total_size + NUM - 1) // NUM
    results = [None] * NUM

    def _dl_chunk(idx, start, end):
        try:
            with sess.get(url, timeout=(5, timeout), stream=True,
                          headers={**ua, "Range": f"bytes={start}-{end}"}) as resp:
                if resp.status_code not in (200, 206):
                    resp.raise_for_status()
                buf = b""
                for p in resp.iter_content(chunk_size=65536):
                    if not p:
                        break
                    buf += p
                results[idx] = buf
        except Exception as e:
            results[idx] = e

    threads = []
    for i in range(NUM):
        start = i * chunk_size
        end = min(start + chunk_size - 1, total_size - 1)
        t = threading.Thread(target=_dl_chunk, args=(i, start, end), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # 组装并校验 (任一线程失败或尺寸不符则降级重下)
    data = b""
    for i, part in enumerate(results):
        if isinstance(part, Exception):
            log.warning("chunked download failed chunk %d: %s, fallback single", i, part)
            return _download_fallback(url, max_bytes, timeout)
        expected = min(chunk_size, total_size - i * chunk_size)
        if len(part) != expected:
            log.warning("chunked download chunk %d short (%d != %d), fallback single",
                        i, len(part), expected)
            return _download_fallback(url, max_bytes, timeout)
        data += part
    return data


def _download_fallback(url: str, max_bytes: int = MAX_IMAGE_BYTES,
                       timeout: int = DOWNLOAD_TIMEOUT) -> bytes:
    """单连接兜底下载 (无 HEAD/Range)."""
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

def _cache_clear() -> int:
    with _CACHE_LOCK:
        count = len(_RESPONSE_CACHE)
        _RESPONSE_CACHE.clear()
        return count

# DB 持久缓存 (L2): 原始URL->打码后URL 映射, 进程重启不丢
def _db_cache_get(cache_key: str):
    """查询 L2 缓存。返回图片地址和文件名；过期/文件丢失自动清理。"""
    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT output_url, output_file, face_count, expires_at FROM blur_cache WHERE cache_key=?",
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
        return {"output_url": row["output_url"], "output_file": row["output_file"],
                "face_count": row["face_count"] if row["face_count"] is not None else 0}
    except Exception as e:
        log.warning("db_cache_get failed: %s", e)
        return None


def _db_cache_set(cache_key: str, image_url: str, output_url: str,
                  output_file: str, mode: str, face_count: int = 0):
    """写入 L2 缓存"""
    try:
        now = _utc_now()
        ttl_hours = _get_int_setting("image_ttl_hours", IMAGE_TTL_HOURS, 1, 24 * 365)
        expires = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat(timespec="seconds")
        with _db() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO blur_cache
                   (cache_key, image_url, output_url, output_file, mode, face_count, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (cache_key, image_url, output_url, output_file, mode, face_count, now, expires),
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


def _cache_clear_parent(parent_task_id: str) -> tuple[int, int]:
    """清理某个父任务关联的 L1/L2 缓存，不影响其他父任务。"""
    parent_task_id = parent_task_id.strip()
    if not parent_task_id:
        return 0, 0

    names: set[str] = set()
    urls: set[str] = set()
    with _db() as conn:
        refs = conn.execute(
            """SELECT output_file, output_url FROM requests
               WHERE parent_task_id=? AND (output_file IS NOT NULL OR output_url IS NOT NULL)""",
            (parent_task_id,),
        ).fetchall()
        for row in refs:
            if row["output_file"]:
                names.add(Path(row["output_file"]).name)
            if row["output_url"]:
                urls.add(str(row["output_url"]))

        cache_rows = []
        if names or urls:
            clauses = []
            params: list[str] = []
            if names:
                placeholders = ",".join("?" for _ in names)
                clauses.append(f"output_file IN ({placeholders})")
                params.extend(sorted(names))
            if urls:
                placeholders = ",".join("?" for _ in urls)
                clauses.append(f"output_url IN ({placeholders})")
                params.extend(sorted(urls))
            cache_rows = conn.execute(
                "SELECT cache_key FROM blur_cache WHERE " + " OR ".join(clauses), params
            ).fetchall()
            keys = [row["cache_key"] for row in cache_rows]
            if keys:
                placeholders = ",".join("?" for _ in keys)
                conn.execute(f"DELETE FROM blur_cache WHERE cache_key IN ({placeholders})", keys)
        else:
            keys = []

    def matches(resp: dict) -> bool:
        output_file = resp.get("output_file")
        output_url = resp.get("output_url")
        return (
            (output_file and Path(str(output_file)).name in names)
            or (output_url and str(output_url) in urls)
        )

    l1_count = 0
    with _CACHE_LOCK:
        for key, entry in list(_RESPONSE_CACHE.items()):
            if matches(entry.get("resp") or {}):
                del _RESPONSE_CACHE[key]
                l1_count += 1
    return l1_count, len(keys)


def _public_url_for(path: Path, request_base: str) -> str:
    """生成 output_url. 优先用 PUBLIC_BASE_URL, 否则用 request base_url"""
    rel = path.name
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL.rstrip('/')}/static/{rel}"
    return f"{request_base}static/{rel}"


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.get("/lab")
def lab_page(request: Request):
    """打码实验室 - 独立测试页面"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>打码实验室 · FaceBlur</title>    
    <!-- ── 打码实验室独立页面 ── -->
    <style>
    :root {
      --ink: #171717; --muted: #6f6b62; --bg: #f4f2ec;
      --panel: #fffdf7; --line: #d8d2c4; --accent: #1f7a5a; --warn: #a84526;
      --accent-light: #e8f1eb; --text: var(--ink); --text-dim: var(--muted); --bg-card: var(--panel);
    }
    [data-theme="dark"] {
      --ink: #e0e0e0; --muted: #999; --bg: #1a1a1a; --panel: #2a2a2a;
      --line: #444; --accent: #5cbf90; --warn: #e07b5a; --accent-light: #1a2a22;
    }
    * { box-sizing: border-box; }
    body { margin:0; background: var(--bg); color: var(--ink); font-family: system-ui,-apple-system,sans-serif; -webkit-font-smoothing: antialiased; }
    .lab-wrap { max-width:none; margin:0 auto; padding:28px 100px 40px; }
    .lab-header { display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:20px; padding-bottom:14px; border-bottom:2px solid var(--line); }
    .lab-header h2 { margin:0; font-size:26px; font-weight:700; letter-spacing:-0.3px; }
    .lab-header .lab-badge { font-size:12px; background:var(--accent-light); color:var(--accent); padding:4px 12px; border-radius:99px; font-weight:600; }
    .lab-grid { display:grid; grid-template-columns: minmax(0,1.55fr) minmax(350px,1fr); gap:20px; }
    .lab-card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:22px; box-shadow:0 1px 4px rgba(0,0,0,0.03); }
    [data-theme="dark"] .lab-card { box-shadow:0 1px 8px rgba(0,0,0,0.3); }
    .lab-dropzone { border:2px dashed var(--line); border-radius:16px; padding:28px 24px 22px; text-align:center; cursor:pointer; transition:all .25s; background:var(--panel); }
    .lab-dropzone:hover { border-color:var(--accent); background:var(--accent-light); transform:translateY(-1px); box-shadow:0 4px 16px rgba(0,0,0,0.06); }
    .lab-dropzone input { display:none; }
    .lab-dropzone .dz-icon { font-size:42px; display:block; margin-bottom:10px; }
    .lab-dropzone .dz-title { font-size:15px; font-weight:600; margin-bottom:4px; }
    .lab-dropzone .dz-hint { font-size:12px; color:var(--muted); }
    .lab-url-row { display:flex; gap:10px; margin-top:14px; }
    .lab-url-row input { flex:1; padding:11px 14px; border:1px solid var(--line); border-radius:10px; background:var(--panel); color:var(--ink); font-size:14px; transition:border-color .2s; }
    .lab-url-row input:focus { outline:none; border-color:var(--accent); box-shadow:0 0 0 3px rgba(31,122,90,0.1); }
    .lab-url-row input::placeholder { color:var(--muted); }
    .lab-preview { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:18px; }
    .lab-preview-item { background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden; transition:box-shadow .2s; }
    .lab-preview-item:hover { box-shadow:0 4px 20px rgba(0,0,0,0.08); }
    .lab-preview-item img { display:block; width:100%; aspect-ratio:4/3; min-height:340px; object-fit:cover; cursor:zoom-in; background:var(--bg); }
    .lab-preview-item .pv-label { display:flex; align-items:center; gap:8px; padding:10px 14px; font-size:12px; font-weight:600; color:var(--muted); border-top:1px solid var(--line); }
    .lab-preview-item .pv-label::before { content:''; display:inline-block; width:8px; height:8px; border-radius:2px; }
    .lab-preview-item.before .pv-label::before { background:var(--warn); }
    .lab-preview-item.after .pv-label::before { background:var(--accent); }
    .lab-confidence { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:16px; }
    .lab-conf-widget { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }
    .lab-conf-widget .cw-title { font-size:12px; font-weight:600; color:var(--muted); margin-bottom:8px; text-transform:uppercase; letter-spacing:0.5px; }
    .lab-conf-widget .cw-stat { font-size:28px; font-weight:700; line-height:1; margin-bottom:4px; }
    .lab-conf-widget .cw-stat.danger { color:var(--warn); }
    .lab-conf-widget .cw-stat.safe { color:var(--accent); }
    .lab-conf-widget .cw-detail { font-size:12px; color:var(--muted); line-height:1.6; }
    .lab-conf-widget .cw-bar { height:6px; border-radius:3px; background:var(--line); margin-top:10px; overflow:hidden; }
    .lab-conf-widget .cw-bar-fill { height:100%; border-radius:3px; transition:width .4s ease; }
    .cw-bar-fill.danger { background:var(--warn); }
    .cw-bar-fill.safe { background:var(--accent); }
    .lab-params { background:var(--panel); border:1px solid var(--line); border-radius:14px; box-shadow:0 1px 6px rgba(0,0,0,0.03); overflow:hidden; }
    [data-theme="dark"] .lab-params { box-shadow:0 1px 12px rgba(0,0,0,0.28); }
    .lab-params-head { padding:14px 18px; border-bottom:1px solid var(--line); }
    .lab-params-head h3 { margin:0; font-size:15px; font-weight:600; }
    .lab-params-body { padding:14px 18px; }
    .lab-param { margin-bottom:10px; }
    .lab-param:last-child { margin-bottom:0; }
    .lab-param label { display:flex; justify-content:space-between; align-items:center; font-size:13px; font-weight:500; margin-bottom:2px; color:var(--ink); }
    .lab-param label span { color:var(--ink); font-weight:700; font-size:14px; font-variant-numeric:tabular-nums; background:var(--bg); padding:2px 8px; border-radius:5px; min-width:36px; text-align:center; }
    .lab-param input[type=range] { width:100%; height:6px; accent-color:var(--accent); border-radius:3px; }
    .lab-param select { width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:10px; background:var(--panel); color:var(--ink); font-size:14px; cursor:pointer; }
    .lab-param .hint { font-size:11px; color:var(--muted); margin-top:1px; }
    .lab-btns { display:flex; gap:10px; margin-top:12px; }
    .btn { border:none; padding:12px 20px; border-radius:10px; cursor:pointer; font-size:14px; font-weight:600; transition:all .15s; display:inline-flex; align-items:center; gap:6px; }
    .btn.primary { background:var(--accent); color:#fff; }
    .btn.primary:hover { filter:brightness(1.1); transform:translateY(-1px); box-shadow:0 4px 14px rgba(31,122,90,0.25); }
    .btn.primary:active { transform:translateY(0); }
    .btn.secondary { background:transparent; color:var(--ink); border:1px solid var(--line); }
    .btn.secondary:hover { background:var(--bg); border-color:var(--ink); }
    .btn:disabled { opacity:0.35; cursor:not-allowed; transform:none !important; box-shadow:none !important; }
    [data-theme="dark"] .btn.secondary { border-color:#555; }
    [data-theme="dark"] .btn.secondary:hover { border-color:#888; }
    .lab-presets { border-top:1px solid var(--line); margin-top:12px; padding-top:12px; }
    .lab-presets h4 { margin:0 0 6px; font-size:13px; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; }
    .lab-presets-row select { width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:10px; background:var(--panel); color:var(--ink); font-size:14px; }
    .lab-presets-actions { display:flex; gap:8px; margin-top:6px; }
    .status-text { font-size:13px; color:var(--muted); margin-top:10px; line-height:1.5; padding:10px 14px; background:var(--bg); border-radius:8px; }
    .lab-image-modal { position:fixed; inset:0; z-index:30; display:grid; place-items:center; padding:24px; background:rgba(0,0,0,.88); backdrop-filter:blur(4px); }
    .lab-image-modal[hidden] { display:none; }
    .lab-image-modal img { max-width:94vw; max-height:88vh; border-radius:12px; box-shadow:0 8px 40px rgba(0,0,0,0.4); transform-origin: center center; touch-action: none; }
    .lab-image-modal-close { position:fixed; top:16px; right:20px; border:0; background:rgba(255,255,255,0.12); color:white; width:40px; height:40px; border-radius:8px; font-size:22px; cursor:pointer; transition:background .15s; }
    .lab-image-modal-close:hover { background:rgba(255,255,255,0.22); }
    @media (max-width:768px) { .lab-grid,.lab-preview,.lab-confidence { grid-template-columns:1fr; } .lab-wrap { padding:24px 16px 40px; } .lab-header { flex-direction:column; align-items:flex-start; gap:10px; } }
    .spinner { display:inline-block; width:16px; height:16px; border:2px solid rgba(255,255,255,0.3); border-top-color:#fff; border-radius:50%; animation:spin .6s linear infinite; }
    @keyframes spin { to { transform:rotate(360deg); } }
    </style>
</head>
<body>
    <div class="lab-wrap">
      <div class="lab-header">
        <h2>🧪 打码实验室</h2>
        <span class="lab-badge">独立测试环境</span>
      </div>
      <div class="lab-grid">
        <div>
          <div class="lab-dropzone" onclick="document.getElementById('lab-file').click()" id="lab-drop">
            <span class="dz-icon">🖼️</span>
            <div class="dz-title">点击上传图片 或拖拽到此处</div>
            <div class="dz-hint">JPG / PNG / WebP · 最大 20MB</div>
            <input type="file" id="lab-file" accept="image/*" onchange="labUpload(this)" />
          </div>
          <div class="lab-url-row">
            <input id="lab-url" type="url" placeholder="🔗  或粘贴图片 URL..." onkeydown="if(event.key==='Enter')labTest()" />
          </div>
          <div class="lab-preview" id="lab-preview">
            <div class="lab-preview-item before">
              <img id="lab-before" style="display:none" onclick="labOpenPreview(this)" />
              <div class="pv-label">📷 原图</div>
            </div>
            <div class="lab-preview-item after">
              <img id="lab-after" style="display:none" onclick="labOpenPreview(this)" />
              <div class="pv-label">✨ 打码结果<span id="lab-face-info"></span></div>
            </div>
          </div>
          <div class="lab-confidence" id="lab-confidence" hidden>
            <div class="lab-conf-widget">
              <div class="cw-title">原图人脸检测</div>
              <div class="cw-stat danger" id="lab-conf-before-stat">-</div>
              <div class="cw-detail" id="lab-confidence-before">-</div>
              <div class="cw-bar"><div class="cw-bar-fill danger" id="lab-conf-before-bar" style="width:0%"></div></div>
            </div>
            <div class="lab-conf-widget">
              <div class="cw-title">打码后人脸检测</div>
              <div class="cw-stat safe" id="lab-conf-after-stat">-</div>
              <div class="cw-detail" id="lab-confidence-after">-</div>
              <div class="cw-bar"><div class="cw-bar-fill safe" id="lab-conf-after-bar" style="width:0%"></div></div>
            </div>
          </div>
        </div>
        <div class="lab-params">
          <div class="lab-params-head"><h3>⚙️ 打码参数</h3></div>
          <div class="lab-params-body">
            <div class="lab-param">
              <label>打码模式</label>
              <select id="lab-mode" multiple size="5" onchange="labToggleMode()">
                <option value="landmark_whole_face">整脸红点遮罩</option>
                <option value="landmark">关键点遮罩</option>
                <option value="pixelate">马赛克</option>
                <option value="gaussian">高斯模糊</option>
                <option value="solid">纯色遮挡</option>
              </select>
              <div class="hint">按住 Ctrl/Command 可多选，执行时按选中顺序叠加</div>
            </div>
            <div class="lab-param"><label>距离分档方案 JSON</label><textarea id="lab-profiles" rows="8" style="width:100%;font:12px ui-monospace,monospace;padding:8px;border:1px solid var(--line);border-radius:8px">[{"name":"small","min_width":0,"max_width":99,"modes":["landmark_whole_face"],"face_grid_step":20,"dot_radius":1,"grid_n":3},{"name":"medium","min_width":100,"max_width":199,"modes":["landmark_whole_face"],"face_grid_step":12,"dot_radius":2,"grid_n":5},{"name":"large","min_width":200,"max_width":10000,"modes":["landmark_whole_face"],"face_grid_step":14,"dot_radius":3,"grid_n":5}]</textarea><div class="hint">每档按原始脸宽命中，可为每档配置多个 modes</div></div>
            <div class="lab-param"><label>检测阈值 <span id="lab-score-v">0.52</span></label><input type="range" id="lab-score" min="0.3" max="1.0" step="0.01" value="0.52" oninput="document.getElementById('lab-score-v').textContent=Number(this.value).toFixed(2)" /><div class="hint">越小越灵敏，可能有误检</div></div>
            <div class="lab-param"><label>扩框比例 <span id="lab-expand-v">0.30</span></label><input type="range" id="lab-expand" min="0" max="1.0" step="0.05" value="0.30" oninput="document.getElementById('lab-expand-v').textContent=Number(this.value).toFixed(2)" /><div class="hint">检测框向外扩展的安全余量</div></div>
            <div class="lab-param"><label>跳小脸(px) <span id="lab-minface-v">50</span></label><input type="range" id="lab-minface" min="0" max="500" step="5" value="50" oninput="document.getElementById('lab-minface-v').textContent=this.value" /><div class="hint">脸宽小于此值直接跳过，0=全部打码</div></div>
            <div class="lab-param"><label>网格间距 <span id="lab-step-v">14</span></label><input type="range" id="lab-step" min="4" max="40" step="1" value="14" oninput="document.getElementById('lab-step-v').textContent=this.value; labStepChanged()" /><div class="hint">越小越密集</div></div>
            <div class="lab-param"><label>红点半径 <span id="lab-dot-v">3</span></label><input type="range" id="lab-dot" min="1" max="10" step="1" value="3" oninput="document.getElementById('lab-dot-v').textContent=this.value" /><div class="hint">红点大小(px)</div></div>
            <div class="lab-param"><label>网格密度N <span id="lab-n-v">5</span></label><input type="range" id="lab-n" min="1" max="9" step="1" value="5" oninput="document.getElementById('lab-n-v').textContent=this.value" /><div class="hint">关键点周围叠加层数</div></div>
            <div class="lab-btns">
              <button class="btn primary" onclick="labTest()" id="lab-test-btn">🚀 执行打码</button>
              <button class="btn secondary" onclick="labSyncGlobal()" id="lab-sync-btn" disabled>📋 同步到全局</button>
            </div>
            <div class="lab-presets">
              <h4>自定义预设</h4>
              <div class="lab-presets-row">
                <select id="lab-preset" onchange="labApplyPreset(this.value)"><option value="">选择预设...</option></select>
              </div>
              <div class="lab-presets-actions">
                <button class="btn secondary" onclick="labSavePreset()">💾 保存当前</button>
                <button class="btn secondary" onclick="labDeletePreset()" id="lab-delete-preset" disabled>🗑 删除</button>
                <button class="btn secondary" onclick="labResetDefaults()">↩ 恢复默认</button>
              </div>
            </div>
            <p id="lab-status" class="status-text"></p>
          </div>
        </div>
      </div>
    </div>
    <div class="lab-image-modal" id="lab-image-modal" hidden onclick="labClosePreview(event)">
      <button class="lab-image-modal-close" type="button" aria-label="关闭图片预览" onclick="labClosePreview()">&times;</button>
      <img id="lab-image-modal-content" alt="实验室图片大图预览" />
    </div>
<script>
(function(){
  try {
    const s = localStorage.getItem('faceblur_theme');
    if(s === 'dark') document.documentElement.setAttribute('data-theme','dark');
  } catch(_){}
})();
const BASE = window.location.origin;
const LAB_TOKEN = new URLSearchParams(location.search).get("token") || "";
const LAB_PRESETS_KEY = "faceblur.lab.presets.v1";
const LAB_DEFAULTS = {mode:"landmark_whole_face", modes:["landmark_whole_face"], face_profiles:[], score_threshold:0.52, expand_ratio:0.30, min_face_skip:40, face_grid_step:14, dot_radius:3, grid_n:5};
async function apiLab(path, opts={}){
  if(LAB_TOKEN){ opts.headers = opts.headers || {}; opts.headers["X-Admin-Token"] = LAB_TOKEN; }
  const r = await fetch(BASE+path, opts);
  const t = await r.text();
  let data;
  try { data = JSON.parse(t); } catch(e) { throw new Error(t || `请求失败 (${r.status})`); }
  if(!r.ok){ throw new Error(data.detail || data.message || `请求失败 (${r.status})`); }
  return data;
}
function labFileToBase64(file){
  return new Promise((ok,err)=>{
    const r = new FileReader();
    r.onload = () => ok(r.result.split(",")[1]);
    r.onerror = err;
    r.readAsDataURL(file);
  });
}
async function labUpload(input){
  const f = input.files[0];
  if(!f) return;
  if(f.size > 20*1024*1024){ alert("图片不能超过 20MB"); return; }
  document.getElementById("lab-status").textContent = "正在加载图片...";
  const b64 = await labFileToBase64(f);
  document.getElementById("lab-base64").value = b64;
  const u = URL.createObjectURL(f);
  const bf = document.getElementById("lab-before");
  bf.src = u; bf.style.display = "";
  document.getElementById("lab-after").style.display = "none";
  document.getElementById("lab-confidence").hidden = true;
  document.getElementById("lab-status").textContent = "图片已就绪, 点击「执行打码」";
  document.getElementById("lab-url").value = "";
  document.getElementById("lab-sync-btn").disabled = true;
}
function labToggleMode(){
  const m = Array.from(document.getElementById("lab-mode").selectedOptions).map(x=>x.value);
  const lm = m.some(x=>x.startsWith("landmark"));
  ["lab-step","lab-dot","lab-n"].forEach(id=>document.getElementById(id).parentElement.style.display = lm?"":"none");
}
function labReadPresets(){
  try { const value = JSON.parse(localStorage.getItem(LAB_PRESETS_KEY) || "{}"); return value && typeof value === "object" ? value : {}; }
  catch(e) { return {}; }
}
function labWritePresets(presets){
  try { localStorage.setItem(LAB_PRESETS_KEY, JSON.stringify(presets)); return true; }
  catch(e) { document.getElementById("lab-status").textContent = "✗ 预设保存失败: 浏览器存储不可用"; return false; }
}
function labCurrentParams(){
  let face_profiles=[]; try { face_profiles=JSON.parse(document.getElementById("lab-profiles").value||"[]"); } catch(e) { throw new Error("距离分档 JSON 格式错误"); }
  const modes=Array.from(document.getElementById("lab-mode").selectedOptions).map(x=>x.value);
  return {mode:modes[0]||"gaussian", modes, face_profiles, score_threshold:Number(document.getElementById("lab-score").value), expand_ratio:Number(document.getElementById("lab-expand").value), min_face_skip:Number(document.getElementById("lab-minface").value), face_grid_step:Number(document.getElementById("lab-step").value), dot_radius:Number(document.getElementById("lab-dot").value), grid_n:Number(document.getElementById("lab-n").value)};
}
function labSetParams(params){
  const values = Object.assign({}, LAB_DEFAULTS, params || {});
  ["mode","score_threshold","expand_ratio","min_face_skip","face_grid_step","dot_radius","grid_n"].forEach(name => {
    const id = {mode:"lab-mode", score_threshold:"lab-score", expand_ratio:"lab-expand", min_face_skip:"lab-minface", face_grid_step:"lab-step", dot_radius:"lab-dot", grid_n:"lab-n"}[name];
    if(name !== "mode") document.getElementById(id).value = values[name];
  });
  const modes=values.modes||[values.mode]; document.querySelectorAll('#lab-mode option').forEach(o=>o.selected=modes.includes(o.value));
  document.getElementById('lab-profiles').value=JSON.stringify(values.face_profiles||[],null,2);
  document.getElementById("lab-score-v").textContent = Number(values.score_threshold).toFixed(2);
  document.getElementById("lab-expand-v").textContent = Number(values.expand_ratio).toFixed(2);
  document.getElementById("lab-minface-v").textContent = values.min_face_skip;
  document.getElementById("lab-step-v").textContent = values.face_grid_step;
  document.getElementById("lab-dot-v").textContent = values.dot_radius;
  document.getElementById("lab-n-v").textContent = values.grid_n;
  labToggleMode();
  document.getElementById("lab-sync-btn").disabled = true;
}
function labEscapeHtml(value){
  return String(value).replace(/[&<>\"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'\"':"&quot;", "'":"&#39;"}[ch]));
}
function labRefreshPresetOptions(selected=""){
  const select = document.getElementById("lab-preset");
  const presets = labReadPresets();
  select.innerHTML = '<option value="">选择预设...</option>' + Object.keys(presets).sort((a,b)=>a.localeCompare(b,"zh-CN")).map(name => '<option value="' + labEscapeHtml(name) + '">' + labEscapeHtml(name) + '</option>').join("");
  select.value = selected;
  document.getElementById("lab-delete-preset").disabled = !select.value;
}
function labApplyPreset(name){
  if(!name) { document.getElementById("lab-delete-preset").disabled = true; return; }
  const preset = labReadPresets()[name];
  if(preset) { labSetParams(preset); document.getElementById("lab-status").textContent = "✓ 已应用预设: " + name; }
  document.getElementById("lab-delete-preset").disabled = !preset;
}
function labSavePreset(){
  const name = prompt("请输入预设名称");
  if(!name || !name.trim()) return;
  const cleanName = name.trim();
  const presets = labReadPresets();
  if(presets[cleanName] && !confirm("预设已存在，是否覆盖？")) return;
  presets[cleanName] = labCurrentParams();
  if(!labWritePresets(presets)) return;
  labRefreshPresetOptions(cleanName);
  document.getElementById("lab-status").textContent = "✓ 已保存预设: " + cleanName;
}
function labDeletePreset(){
  const select = document.getElementById("lab-preset");
  const name = select.value;
  if(!name) return;
  if(!confirm("确定删除预设「" + name + "」？")) return;
  const presets = labReadPresets();
  delete presets[name];
  if(!labWritePresets(presets)) return;
  labRefreshPresetOptions();
  document.getElementById("lab-status").textContent = "✓ 已删除预设: " + name;
}
function labResetDefaults(){
  labSetParams(LAB_DEFAULTS);
  labRefreshPresetOptions();
  document.getElementById("lab-status").textContent = "✓ 已恢复默认参数";
}
async function labTest(){
  let params; try { params=labCurrentParams(); } catch(e) { document.getElementById("lab-status").textContent="✗ "+e.message; return; }
  const m=params.modes[0]||"gaussian";
  const b64 = document.getElementById("lab-base64").value;
  const url = document.getElementById("lab-url").value.trim();
  if(!b64 && !url){ alert("请先上传图片或输入 URL"); return; }
  const btn = document.getElementById("lab-test-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>处理中...';
  document.getElementById("lab-status").textContent = "正在提交打码请求...";
  document.getElementById("lab-confidence").hidden = true;
  try {
    const body = {mode: m, modes: params.modes, face_profiles: params.face_profiles,
      score_threshold: Number(document.getElementById("lab-score").value),
      expand_ratio: Number(document.getElementById("lab-expand").value),
      min_face_skip: Number(document.getElementById("lab-minface").value),
      face_grid_step: Number(document.getElementById("lab-step").value),
      dot_radius: Number(document.getElementById("lab-dot").value),
      grid_n: Number(document.getElementById("lab-n").value),
    };
    if(b64) body.image_base64 = b64;
    if(url) body.image_url = url;
    const d = await apiLab("/api/lab/test", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
    const af = document.getElementById("lab-after");
    af.src = "data:image/jpeg;base64," + d.output_base64;
    af.style.display = "";
    document.getElementById("lab-face-info").textContent = " (" + d.face_count + "张脸, " + (d.elapsed_ms||0).toFixed(0) + "ms)";
    const updateConfWidget = (prefix, value) => {
      const avgPct = (value.avg_score * 100).toFixed(1);
      const elStat = document.getElementById('lab-conf-' + prefix + '-stat');
      const elDetail = document.getElementById('lab-confidence-' + prefix);
      const elBar = document.getElementById('lab-conf-' + prefix + '-bar');
      if(elStat) elStat.textContent = value.face_count + ' 张脸';
      if(elDetail) elDetail.textContent = '最高 ' + (value.max_score * 100).toFixed(1) + '% · 平均 ' + avgPct + '%' + (value.scores && value.scores.length ? ' · 明细 ' + value.scores.map(s => (s*100).toFixed(1) + '%').join('、') : '');
      if(elBar) elBar.style.width = avgPct + '%';
    };
    updateConfWidget('before', d.confidence.before);
    updateConfWidget('after', d.confidence.after);
    document.getElementById("lab-confidence").hidden = false;
    document.getElementById("lab-status").textContent = "✓ 完成";
    document.getElementById("lab-sync-btn").disabled = false;
    if(!b64 && url){
      const bf = document.getElementById("lab-before");
      bf.src = url; bf.style.display = "";
    }
  } catch(e) {
    document.getElementById("lab-status").textContent = "✗ 错误: " + e.message;
  }
  btn.disabled = false;
  btn.innerHTML = "🧪 执行打码";
}
async function labSyncGlobal(){
  if(!confirm("将当前参数同步到全局设置？\\n(对后续新建请求生效)")) return;
  let params; try { params=labCurrentParams(); } catch(e) { document.getElementById("lab-status").textContent="✗ "+e.message; return; }
  const body = {
    score_threshold: Number(document.getElementById("lab-score").value),
    expand_ratio: Number(document.getElementById("lab-expand").value),
    min_face_skip: Number(document.getElementById("lab-minface").value),
    face_grid_step: Number(document.getElementById("lab-step").value),
    dot_radius: Number(document.getElementById("lab-dot").value),
    grid_n: Number(document.getElementById("lab-n").value), face_profiles: params.face_profiles,
  };
  try {
    await apiLab("/api/admin/settings", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
    document.getElementById("lab-sync-btn").disabled = true;
    document.getElementById("lab-status").textContent = "✓ 已同步到全局设置";
  } catch(e) {
    document.getElementById("lab-status").textContent = "✗ 同步失败: " + e.message;
  }
}
function labStepChanged(){
  document.getElementById("lab-step").value = document.getElementById("lab-step").value;
}
function labOpenPreview(image){
  if(!image || !image.src) return;
  const modal = document.getElementById("lab-image-modal");
  const preview = document.getElementById("lab-image-modal-content");
  preview.src = image.src;
  preview.alt = image.alt || "实验室图片大图预览";
  modal.hidden = false;

  let labZoom = 1, labTX = 0, labTY = 0;
  let labDragging = false, labDownX = 0, labDownY = 0, labStartTX = 0, labStartTY = 0;

  function clamp(){
    const w = (preview.naturalWidth || preview.width) * labZoom;
    const h = (preview.naturalHeight || preview.height) * labZoom;
    const mw = modal.clientWidth, mh = modal.clientHeight;
    const maxTX = Math.max(0, (w - mw) / 2 / labZoom);
    const maxTY = Math.max(0, (h - mh) / 2 / labZoom);
    labTX = Math.min(maxTX, Math.max(-maxTX, labTX));
    labTY = Math.min(maxTY, Math.max(-maxTY, labTY));
  }
  function apply(){
    preview.style.transform = `scale(${labZoom}) translate(${labTX}px, ${labTY}px)`;
    preview.style.transition = labDragging ? 'none' : 'transform .15s ease';
    preview.style.cursor = labZoom > 1 ? (labDragging ? 'grabbing' : 'grab') : 'default';
  }
  preview.onload = function(){ labZoom=1; labTX=labTY=0; apply(); };
  labZoom=1; labTX=labTY=0; apply();

  modal._labZoomHandler = function(e){
    e.preventDefault();
    labZoom += e.deltaY < 0 ? 0.1 : -0.1;
    labZoom = Math.min(5, Math.max(0.3, labZoom));
    clamp(); apply();
  };
  modal.addEventListener('wheel', modal._labZoomHandler, {passive:false});

  modal._labDragStart = function(e){
    if(labZoom <= 1) return;
    labDragging = true;
    labDownX = (e.touches ? e.touches[0].clientX : e.clientX);
    labDownY = (e.touches ? e.touches[0].clientY : e.clientY);
    labStartTX = labTX; labStartTY = labTY;
    apply(); e.preventDefault();
  };
  modal._labDragMove = function(e){
    if(!labDragging) return;
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    labTX = labStartTX + (cx - labDownX) / labZoom;
    labTY = labStartTY + (cy - labDownY) / labZoom;
    clamp(); apply();
  };
  modal._labDragEnd = function(){
    labDragging = false; apply();
  };
  preview.addEventListener('mousedown', modal._labDragStart);
  preview.addEventListener('touchstart', modal._labDragStart, {passive:false});
  window.addEventListener('mousemove', modal._labDragMove);
  window.addEventListener('touchmove', modal._labDragMove, {passive:false});
  window.addEventListener('mouseup', modal._labDragEnd);
  window.addEventListener('touchend', modal._labDragEnd);
}
function labClosePreview(event){
  if(event && event.target !== event.currentTarget) return;
  const modal = document.getElementById("lab-image-modal");
  modal.hidden = true;
  const preview = document.getElementById("lab-image-modal-content");
  modal.removeEventListener('wheel', modal._labZoomHandler);
  preview.removeEventListener('mousedown', modal._labDragStart);
  preview.removeEventListener('touchstart', modal._labDragStart);
  window.removeEventListener('mousemove', modal._labDragMove);
  window.removeEventListener('touchmove', modal._labDragMove);
  window.removeEventListener('mouseup', modal._labDragEnd);
  window.removeEventListener('touchend', modal._labDragEnd);
  delete modal._labZoomHandler;
  delete modal._labDragStart; delete modal._labDragMove; delete modal._labDragEnd;
  preview.src = "";
}
// 页面内容在 head 之后才创建，所有 DOM 绑定必须延后到 DOMContentLoaded。
window.addEventListener("DOMContentLoaded", () => {
  const drop = document.getElementById("lab-drop");
  if(drop){
    drop.addEventListener("dragover", e=>{e.preventDefault(); drop.style.borderColor="var(--accent)";});
    drop.addEventListener("dragleave", ()=>{drop.style.borderColor="var(--line)";});
    drop.addEventListener("drop", e=>{
      e.preventDefault(); drop.style.borderColor="var(--line)";
      const f = e.dataTransfer.files[0];
      if(f && f.type.startsWith("image/")){
        const dt = new DataTransfer(); dt.items.add(f);
        document.getElementById("lab-file").files = dt.files;
        labUpload(document.getElementById("lab-file"));
      }
    });
  }
  labSetParams(LAB_DEFAULTS);
  labRefreshPresetOptions();
  document.addEventListener("keydown", event => { if(event.key === "Escape") labClosePreview(); });
});
</script>
<input type="hidden" id="lab-base64" />
</body>
</html>
""")

def _confidence_summary(faces: list) -> dict:
    scores = []
    for face in faces:
        try:
            score = float(face.get("score", 0)) if isinstance(face, dict) else float(face.score)
        except (TypeError, ValueError, AttributeError):
            continue
        scores.append(round(max(0.0, min(1.0, score)), 4))
    return {
        "face_count": len(scores),
        "max_score": max(scores, default=0.0),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "scores": scores,
    }


@app.post("/api/lab/test")
def lab_test(req: FaceBlurRequest, request: Request, authorization: str | None = Header(default=None), x_admin_token: str | None = Header(default=None)):
    """打码实验室 - 需要 Admin 鉴权"""
    _require_admin(request, authorization, x_admin_token, token=None)
    import base64
    img_bytes = None
    if req.image_base64:
        try:
            img_bytes = base64.b64decode(req.image_base64)
        except Exception:
            raise HTTPException(400, "invalid base64 image")
    elif req.image_url:
        try:
            # 实验室常用于临时验证第三方 CDN 图片。单连接避免 Range 请求被
            # 限流或其中一个分段超时，并给慢源足够的读取时间。
            image_url = str(req.image_url)
            _assert_public_url(image_url)
            img_bytes = _download_fallback(image_url, timeout=max(DOWNLOAD_TIMEOUT, 60))
        except Exception as exc:
            raise HTTPException(400, f"图片 URL 下载失败: {exc}") from exc
    else:
        raise HTTPException(400, "need image_base64 or image_url")
    if len(img_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image too large")
    t0 = time.perf_counter()
    blur_params = {}
    selected_modes = req.modes or [req.mode]
    blur_params["modes"] = selected_modes
    blur_params["face_profiles"] = req.face_profiles
    if req.mode == "landmark_whole_face":
        blur_params.update({
            "adaptive": False, "min_face_skip": req.min_face_skip if req.min_face_skip is not None else 50,
            "dot_radius": req.dot_radius, "face_grid_step": req.face_grid_step,
            "grid_n": req.grid_n or 5, "spacing": req.face_grid_step,
        })
    from face_blur import process_image
    try:
        result = process_image(img_bytes, mode=req.mode, score_threshold=req.score_threshold,
                              expand_ratio=req.expand_ratio, return_faces=True, **blur_params)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    elapsed = (time.perf_counter() - t0) * 1000
    out_b64 = base64.b64encode(result["image_bytes"]).decode("ascii")
    before_confidence = _confidence_summary(result.get("faces", []))
    try:
        after_probe = process_image(
            result["image_bytes"], mode="gaussian",
            score_threshold=req.score_threshold, expand_ratio=0, return_faces=True,
        )
        after_confidence = _confidence_summary(after_probe.get("faces", []))
    except Exception as exc:  # noqa: BLE001
        log.warning("lab post-blur confidence detection failed: %s", exc)
        after_confidence = _confidence_summary([])
    return {
        "ok": True, "face_count": result.get("face_count", 0),
        "elapsed_ms": elapsed, "output_base64": out_b64,
        "confidence": {"threshold": req.score_threshold, "before": before_confidence, "after": after_confidence},
    }


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
    explicit = req.model_fields_set
    effective_params = {
        "score_threshold": req.score_threshold if "score_threshold" in explicit else _get_blur_default("score_threshold", 0.52),
        "expand_ratio": req.expand_ratio if "expand_ratio" in explicit else _get_blur_default("expand_ratio", 0.30),
        "min_face_skip": req.min_face_skip if "min_face_skip" in explicit else int(_get_blur_default("min_face_skip", 50)),
        "dot_radius": req.dot_radius if "dot_radius" in explicit else int(_get_blur_default("dot_radius", 3)),
        "face_grid_step": req.face_grid_step if "face_grid_step" in explicit else int(_get_blur_default("face_grid_step", 14)),
        "grid_n": req.grid_n if "grid_n" in explicit else int(_get_blur_default("grid_n", 5)),
        "modes": req.modes if "modes" in explicit and req.modes else [req.mode],
        "face_profiles": req.face_profiles if "face_profiles" in explicit and req.face_profiles else _get_blur_default("face_profiles", []),
    }
    # E: 计算缓存 key (L1 + L2 共用)
    _cache_payload = {k:v for k,v in req.model_dump(mode="json").items() if k not in ("parent_task_id","callback_url","image_base64")}
    _cache_payload.update(effective_params)
    if req.image_url:
        _cache_payload["cache_epoch"] = _get_setting("cache_epoch", "0")
    _cache_req_json = json.dumps(_cache_payload, sort_keys=True, ensure_ascii=False)
    # Base64 图片没有稳定的 URL 标识，不能按 None 参与缓存，否则不同图片会碰撞。
    _ck = _cache_key(str(req.image_url), _cache_req_json) if req.image_url else ""

    # L1: 内存缓存
    _cached = _cache_get(_ck) if _ck else None
    if _cached is not None:
        log.info("[cache] L1 hit")
        r = {**_cached, "task_id": task_id, "parent_task_id": req.parent_task_id or ""}
        _insert_request({
            "task_id": task_id,
            "status": "ok",
            "mode": req.mode,
            "blocked": _cached.get("blocked", 1),
            "face_count": _cached.get("face_count", -1),
            "elapsed_ms": 0,
            "process_ms": 0,
            "image_url": str(req.image_url),
            "output_url": _cached.get("output_url", ""),
            "output_file": _cached.get("output_file") or _static_name_from_url(_cached.get("output_url")),
            "parent_task_id": req.parent_task_id,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent", "")[:300],
            "request_json": _raw_request_record(request, req.model_dump(mode="json")),
            "response_json": json.dumps(r, ensure_ascii=False),
        })
        return r

    # L2: DB 持久缓存
    _db_url = _db_cache_get(_ck) if _ck else None
    if _db_url is not None:
        # 构造与正常打码一致的响应
        log.info("[cache] L2 hit")
        _cached_output_url = _db_url["output_url"]
        _db_face_count = _db_url.get("face_count", 0)
        r = {
            "task_id": task_id,
            "ok": True,
            "blocked": _db_face_count > 0,
            "face_count": _db_face_count,
            "elapsed_ms": 0,
            "mode": req.mode,
            "output_url": _cached_output_url,
            "original_url": str(req.image_url),
            "parent_task_id": req.parent_task_id or "",
            "cached": True,
        }
        _insert_request({
            "task_id": task_id,
            "status": "ok",
            "mode": req.mode,
            "blocked": 1 if _db_face_count > 0 else 0,
            "face_count": _db_face_count,
            "elapsed_ms": 0,
            "process_ms": 0,
            "image_url": str(req.image_url),
            "output_url": _cached_output_url,
            "output_file": _db_url.get("output_file") or _static_name_from_url(_cached_output_url),
            "parent_task_id": req.parent_task_id,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent", "")[:300],
            "request_json": _raw_request_record(request, req.model_dump(mode="json")),
            "response_json": json.dumps(r, ensure_ascii=False),
        })
        return r

    try:
        with _task_slot():
            return _face_blur_impl(task_id, req, request, cache_key=_ck, effective_params=effective_params)
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


def _face_blur_impl(task_id: str, req: FaceBlurRequest, request: Request, *, cache_key: str = "", effective_params: dict | None = None):
    t0 = time.perf_counter()
    image_url = str(req.image_url)
    request_json = _request_record(req, request)
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent", "")[:300]
    attempts = 1
    max_retries = _get_int_setting("max_retries", MAX_RETRIES, 0, 10)
    params = effective_params or {
        "score_threshold": req.score_threshold,
        "expand_ratio": req.expand_ratio,
        "min_face_skip": req.min_face_skip,
        "dot_radius": req.dot_radius,
        "face_grid_step": req.face_grid_step,
        "grid_n": req.grid_n,
        "modes": req.modes or [req.mode],
        "face_profiles": req.face_profiles,
    }

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
            blur_params["dot_radius"] = params["dot_radius"]
            blur_params["spacing"] = req.spacing if "spacing" in req.model_fields_set else params["face_grid_step"]
        elif req.mode == "landmark_whole_face":
            blur_params["dot_radius"] = params["dot_radius"]
            blur_params["spacing"] = req.spacing if "spacing" in req.model_fields_set else params["face_grid_step"]
            blur_params["face_grid_step"] = params["face_grid_step"]
            blur_params["grid_n"] = params["grid_n"]
        blur_params["modes"] = params.get("modes") or [req.mode]
        blur_params["face_profiles"] = params.get("face_profiles") or []
        result, process_attempts, process_error = _run_with_retries(
            "process_image",
            lambda: process_image(
                img_bytes,
                mode=req.mode,
                score_threshold=params["score_threshold"],
                expand_ratio=params["expand_ratio"],
                return_faces=True,
                adaptive=False, min_face_skip=params["min_face_skip"],
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
            "parent_task_id": req.parent_task_id or "",
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
        if cache_key:
            _cache_set(cache_key, response)
        return response

    # 4. 写入静态目录
    out_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
    out_path = STATIC_DIR / out_name
    out_path.write_bytes(result["image_bytes"])
    _invalidate_storage_stats()
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
        "output_file": out_name,
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
    if cache_key:
        _db_cache_set(cache_key, str(req.image_url), response["output_url"],
                      out_name, req.mode, face_count)
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
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        today = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE created_at >= ?", (cutoff,)
        ).fetchone()[0]
        today_ok = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE status='ok' AND created_at >= ?", (cutoff,)
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
    storage = _storage_stats()
    inflight = _inflight_read()
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
            "file_count": storage["file_count"],
            "bytes": storage["bytes"],
        },
        "settings": {
            "score_threshold": _get_blur_default("score_threshold", 0.52),
            "expand_ratio": _get_blur_default("expand_ratio", 0.30),
            "min_face_skip": _get_blur_default("min_face_skip", 50),
            "dot_radius": _get_blur_default("dot_radius", 3),
            "face_grid_step": _get_blur_default("face_grid_step", 14),
            "grid_n": _get_blur_default("grid_n", 5),
            "face_profiles": _get_blur_default("face_profiles", []),
        },
        "last_cleanup": dict(cleanup) if cleanup else None,
    }


@app.get("/api/admin/requests")
def admin_requests(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=10, le=200),
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
    parent_task_id: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_admin(request, authorization, x_admin_token, token)
    items, total = _static_files(offset=offset, limit=limit, include_task_id=True, parent_task_id=parent_task_id)
    return {"ok": True, "items": items, "total": total, "offset": offset, "limit": limit, "has_more": offset + len(items) < total}


@app.post("/api/admin/clear-cache")
def admin_clear_cache(
    request: Request,
    parent_task_id: str | None = Query(default=None, max_length=200),
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_admin(request, authorization, x_admin_token, token)
    if parent_task_id is not None:
        l1_count, l2_count = _cache_clear_parent(parent_task_id)
        return {
            "ok": True,
            "parent_task_id": parent_task_id,
            "cleared_l1": l1_count,
            "cleared_l2": l2_count,
        }
    with _db() as conn:
        n = conn.execute("SELECT COUNT(*) FROM blur_cache").fetchone()[0]
        conn.execute("DELETE FROM blur_cache")
        conn.commit()
    l1_count = _cache_clear()
    _bump_cache_epoch()
    return {"ok": True, "cleared_l1": l1_count, "cleared_l2": n}



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
    score_threshold: Optional[float] = Field(default=None, ge=0.1, le=0.99)
    expand_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_face_skip: Optional[int] = Field(default=None, ge=0, le=500)
    dot_radius: Optional[int] = Field(default=None, ge=1, le=20)
    face_grid_step: Optional[int] = Field(default=None, ge=4, le=60)
    grid_n: Optional[int] = Field(default=None, ge=1, le=11)
    face_profiles: Optional[list[dict]] = None


@app.get("/api/admin/settings")
def admin_get_settings(
    request: Request,
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_admin(request, authorization, x_admin_token, token)
    inflight = _inflight_read()
    return {
        "ok": True,
        "settings": {
            "max_concurrent_tasks": _get_int_setting("max_concurrent_tasks", DEFAULT_MAX_CONCURRENT_TASKS, 1, 128),
            "max_retries": _get_int_setting("max_retries", MAX_RETRIES, 0, 10),
            "retry_backoff_seconds": _get_float_setting("retry_backoff_seconds", RETRY_BACKOFF_SECONDS, 0.0, 10.0),
            "image_ttl_hours": _get_int_setting("image_ttl_hours", IMAGE_TTL_HOURS, 1, 24 * 365),
            "score_threshold": _get_blur_default("score_threshold", 0.52),
            "expand_ratio": _get_blur_default("expand_ratio", 0.30),
            "min_face_skip": _get_blur_default("min_face_skip", 50),
            "dot_radius": _get_blur_default("dot_radius", 3),
            "face_grid_step": _get_blur_default("face_grid_step", 14),
            "grid_n": _get_blur_default("grid_n", 5),
            "face_profiles": _get_blur_default("face_profiles", []),
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
    saved = _set_settings(values)
    if any(key in values for key in ("score_threshold", "expand_ratio", "min_face_skip", "dot_radius", "face_grid_step", "grid_n", "face_profiles")):
        _cache_clear()
        _bump_cache_epoch()
        with _db() as conn:
            conn.execute("DELETE FROM blur_cache")
    return {"ok": True, "saved": saved}


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
    [data-theme="dark"] {
      --bg: #1a1a1a;
      --panel: #2a2a2a;
      --ink: #e0e0e0;
      --muted: #999;
      --line: #444;
      --accent: #5cbf90;
      --warn: #e07b5a;
      --shadow: 0 12px 30px rgba(0,0,0,0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-serif, Georgia, "Times New Roman", "Microsoft YaHei", serif;
    }
    button, input, select { font: inherit; }
    .shell { max-width: none; margin: 0 auto; padding: 28px 100px 48px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: flex-end; margin-bottom: 22px; }
    h1 { margin: 0; font-size: 34px; line-height: 1; letter-spacing: 0; }
    .sub { margin-top: 8px; color: var(--muted); font-size: 14px; }
    .auth { display: flex; gap: 8px; align-items: center; }
    .auth input { width: 260px; padding: 10px 12px; border: 1px solid var(--line); background: var(--panel); border-radius: 6px; }
    .btn { border: 1px solid var(--ink); background: var(--ink); color: white; padding: 10px 14px; border-radius: 6px; cursor: pointer; }
    .btn.secondary { background: transparent; color: var(--ink); border-color: var(--line); }
    .btn.danger { background: var(--warn); border-color: var(--warn); }
    .btn:disabled { cursor: not-allowed; opacity: .45; }
    [data-theme="dark"] .btn { background: #1a1a1a; color: #fff; }
    [data-theme="dark"] .btn.secondary { background: #333; color: #e0e0e0; }
    [data-theme="dark"] .btn.danger { background: #c0392b; }
    [data-theme="dark"] .card { background: var(--panel); }
    [data-theme="dark"] .panel { background: var(--panel); }
    [data-theme="dark"] .field input { background: var(--panel); color: var(--ink); }
    [data-theme="dark"] .file { background: var(--panel); }
    [data-theme="dark"] .fact { background: var(--panel); }
    [data-theme="dark"] .image-preview { background: var(--panel); }
    [data-theme="dark"] .image-preview img { background: var(--bg); }
    [data-theme="dark"] .json-block { background: var(--panel); }
    [data-theme="dark"] .json-block pre { background: var(--bg); color: var(--ink); }
    [data-theme="dark"] th { background: var(--panel); }
    [data-theme="dark"] td { border-bottom-color: var(--line); }
    [data-theme="dark"] .pill { background: #1a2a22; color: var(--accent); }
    [data-theme="dark"] .pill.err { background: #2a1a18; color: var(--warn); }
    [data-theme="dark"] .gallery-tools select, [data-theme="dark"] .request-tools select, [data-theme="dark"] .task-search input { background: var(--panel); color: var(--ink); }
    [data-theme="dark"] .auth input { background: var(--panel); color: var(--ink); }
    [data-theme="dark"] .image-empty { background: var(--bg); }
    [data-theme="dark"] .image-modal img { background: #1a1a1a; }
    [data-theme="dark"] .pager { border-top-color: var(--line); }
    .tabs { display: flex; gap: 4px; margin-bottom: 18px; border-bottom: 1px solid var(--line); }
    .tab { border: 0; border-bottom: 3px solid transparent; background: transparent; color: var(--muted); padding: 10px 16px; cursor: pointer; }
    .tab.active { border-bottom-color: var(--accent); color: var(--ink); font-weight: 700; }
    .tab-view[hidden] { display: none; }
    .grid { display: flex; gap: 12px; margin-bottom: 18px; overflow-x: auto; flex-wrap: nowrap; }
    .grid .card { flex: 0 0 calc(16.66% - 10px); min-width: 150px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: var(--shadow); min-height: 104px; }
    .label { color: var(--muted); font-size: 13px; }
    .metric { margin-top: 8px; font-size: 30px; font-weight: 700; }
    .split { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(360px, 1fr); gap: 18px; }
    .split .files { overflow-y: auto; max-height: 80vh; grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .split .file img { max-height: 130px; }
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
    .file img { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: cover; background: #eee; cursor: zoom-in; }
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
    .image-preview img { display: block; width: 100%; height: 320px; object-fit: contain; background: #efede7; cursor: zoom-in; }
    .image-empty { display: grid; height: 320px; place-items: center; color: var(--muted); }
    .task-json { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .json-block { min-width: 0; border: 1px solid var(--line); background: white; }
    .json-block pre { min-height: 220px; max-height: 520px; margin: 0; padding: 13px; overflow: auto; background: #f7f4ec; white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.55; }
    .image-modal { position: fixed; inset: 0; z-index: 20; display: grid; place-items: center; padding: 28px; background: rgba(0, 0, 0, .78); }
    .image-modal[hidden] { display: none; }
    .image-modal img { display: block; max-width: min(96vw, 1800px); max-height: 90vh; object-fit: contain; background: #111; transform-origin: center center; touch-action: none; }
    .image-modal-close { position: fixed; top: 14px; right: 18px; border: 0; background: transparent; color: white; font-size: 32px; line-height: 1; cursor: pointer; }
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
      <div class="search-center" style="flex:1;display:flex;justify-content:center;align-items:center;gap:8px">
        <input id="task-search" type="search" placeholder="输入任务 ID 定位..." aria-label="任务 ID" style="width:240px;padding:9px 12px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);font-size:13px" />
        <button class="btn secondary" onclick="findTask()" style="white-space:nowrap">查询</button>
        <button class="btn danger" onclick="clearParentCache()" style="white-space:nowrap" title="按当前父任务 ID 清除缓存">清除父任务缓存</button>
      </div>
      <div class="auth">
        <input id="token" type="password" placeholder="Admin token" autocomplete="off" />
        <button class="btn" onclick="saveToken()">保存</button>
        <button class="btn secondary" onclick="loadAll()">刷新</button>
        <button id="themeToggle" class="btn secondary" title="切换深色模式" onclick="toggleTheme()" style="font-size:16px">🌙</button>
      </div>
    </header>

    <nav class="tabs" aria-label="管理页面标签">
      <button class="tab active" data-tab="overview" onclick="showTab('overview')">概览</button>
      <button class="tab" data-tab="gallery" onclick="showGalleryTab()">图片库</button>
      <button class="tab" id="task-tab" data-tab="task" onclick="showTab('task')" hidden>任务详情</button>
      <button class="tab" data-tab="settings" onclick="showTab('settings'); loadSettingsTab();">⚙ 全局设置</button>
      <button class="tab" data-tab="lab" onclick="showLabTab()">🧪 实验室</button>
    </nav>

    <div class="status" id="status">正在加载...</div>
    <div class="tab-view" data-tab="overview-stats">
    <section class="grid">
      <div class="card"><div class="label">总请求</div><div class="metric" id="m-total">-</div></div>
      <div class="card"><div class="label">已打码</div><div class="metric" id="m-blocked">-</div></div>
      <div class="card"><div class="label">成功率</div><div class="metric" id="m-rate">-</div></div>
      <div class="card"><div class="label">24h 成功率</div><div class="metric" id="m-rate-24h">-</div></div>
      <div class="card"><div class="label">静态图片</div><div class="metric" id="m-files">-</div></div>
      <div class="card" style="border-color:var(--accent)"><div class="label">实时并发</div><div class="metric" id="m-inflight">-</div></div>
    </section>
    </div>

<div class="tab-view" data-tab="settings" hidden>
    <section class="panel" style="margin-bottom:14px">
      <div class="panel-head"><h2>⚙ 全局设置</h2><button class="btn primary" onclick="saveSettings()">保存设置</button> <button class="btn secondary" onclick="clearCache()">🗑 清空缓存</button></div>
      <div class="settings" style="gap:14px">
        <div class="field" title="允许同时处理的最大并发请求数"><label>并行上限</label><input id="s-concurrency" type="number" min="1" max="128"/></div>
        <div class="field" title="请求失败后的最大重试次数"><label>重试次数</label><input id="s-retries" type="number" min="0" max="10"/></div>
        <div class="field" title="重试时每次等待的秒数增量"><label>退避秒数</label><input id="s-backoff" type="number" min="0" max="10" step="0.1"/></div>
        <div class="field" title="静态图片的保留时长"><label>保留小时</label><input id="s-ttl" type="number" min="1" max="8760"/></div>
      </div>
    </section>
    <section class="panel" style="margin-bottom:14px">
      <div class="panel-head"><h2>🎨 打码全局默认参数</h2><span style="color:var(--text-dim);font-size:13px">调整后对新建请求生效（已缓存图不受影响）</span></div>
      <div class="settings" style="gap:14px">
        <div class="field" title="人脸检测置信度阈值(0.3-1.0) 越小越灵敏但误检越多"><label>检测阈值</label><input id="s-score" type="number" min="0.3" max="1.0" step="0.01"/></div>
        <div class="field" title="扩框比例(0-1.0) 检测框向外扩展的安全余量"><label>扩框比例</label><input id="s-expand" type="number" min="0" max="1.0" step="0.05"/></div>
        <div class="field" title="极小脸(脸宽<此值px)直接跳过不打码 0=全都打"><label>跳小脸(px)</label><input id="s-minface" type="number" min="0" max="500"/></div>
        <div class="field" title="红点半径(px) 大脸固定3px 中脸2px 小脸1px"><label>红点半径</label><input id="s-dot" type="number" min="1" max="10"/></div>
        <div class="field" title="打码网格间距(px) 越小越密 大脸14/中脸12/小脸20"><label>网格间距</label><input id="s-step" type="number" min="4" max="40"/></div>
      </div>
    </section>
    <p id="settings-note" style="color:var(--text-dim);font-size:13px;margin:0 0 14px 0">提示: 打码参数中"检测阈值"和"扩框比例"对非 landmark 模式也生效</p>
    </div>

    <div class="tab-view" data-tab="overview">
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
    <div class="tab-view" data-tab="lab" hidden>
      <section class="panel"><iframe id="lab-frame" title="打码实验室" style="display:block;width:100%;height:900px;border:0"></iframe></section>
    </div>
  </main>
  <div class="image-modal" id="image-modal" hidden onclick="closeImagePreview(event)">
    <button class="image-modal-close" type="button" aria-label="关闭图片预览" onclick="closeImagePreview()">&times;</button>
    <img id="image-modal-content" alt="图片大图预览" />
  </div>
  <script>
    const tokenEl = document.getElementById('token');
    let activeTab = 'overview';
    let activeParentSearch = '';
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
    function setTheme(dark){
      document.documentElement.setAttribute('data-theme', dark ? 'dark' : '');
      try { localStorage.setItem('faceblur_theme', dark ? 'dark' : 'light'); } catch(_){}
      const btn = document.getElementById('themeToggle');
      if(btn) btn.textContent = dark ? '☀️' : '🌙';
    }
    function toggleTheme(){
      setTheme(document.documentElement.getAttribute('data-theme') !== 'dark');
    }
    (function(){
      try {
        const s = localStorage.getItem('faceblur_theme');
        if(s === 'dark') setTheme(true);
        else if(s === 'light' || !s) setTheme(false);
      } catch(_){}
    })();
    function headers(){ const t = tokenEl.value.trim(); return t ? {'X-Admin-Token': t} : {}; }
    function setStatus(text){ document.getElementById('status').textContent = text; }
    function fmtBytes(n){ if(!n) return '0 B'; const u=['B','KB','MB','GB']; let i=0; while(n>=1024&&i<u.length-1){n/=1024;i++;} return `${n.toFixed(i?1:0)} ${u[i]}`; }
    function fmtNum(n){
      n = Number(n) || 0;
      if(n >= 1e6) return (n/1e6).toFixed(2) + '百万';
      if(n >= 1e5) return (n/1e5).toFixed(2) + '十万';
      if(n >= 1e4) return (n/1e4).toFixed(2) + '万';
      if(n >= 1e3) return (n/1e3).toFixed(2) + '千';
      return n.toString();
    }
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
      const _t = document.getElementById('m-total');
      _t.textContent = fmtNum(d.requests.total);
      _t.title = d.requests.total.toLocaleString() + ' 次请求';
      const _b = document.getElementById('m-blocked');
      _b.textContent = fmtNum(d.requests.blocked);
      _b.title = d.requests.blocked.toLocaleString() + ' 张已打码';
      const _r = document.getElementById('m-rate');
      _r.textContent = d.requests.success_rate.toFixed(2) + '%';
      _r.title = d.requests.success_rate + '% (累计)';
      const _r24 = document.getElementById('m-rate-24h');
      _r24.textContent = d.requests.success_rate_24h.toFixed(2) + '%';
      _r24.title = d.requests.success_rate_24h + '% (24h)';
      const _f = document.getElementById('m-files');
      _f.textContent = fmtNum(d.storage.file_count);
      _f.title = d.storage.file_count.toLocaleString() + ' 张静态图片 (' + fmtBytes(d.storage.bytes) + ')';
      const _i = document.getElementById('m-inflight');
      _i.textContent = d.service.inflight_tasks + '/' + d.service.max_concurrent_tasks;
      _i.title = '当前 ' + d.service.inflight_tasks + ' / 上限 ' + d.service.max_concurrent_tasks + ' 并发';
      document.getElementById('s-concurrency').value = d.service.max_concurrent_tasks;
      document.getElementById('s-retries').value = d.service.max_retries;
      document.getElementById('s-backoff').value = d.service.retry_backoff_seconds;
      document.getElementById('s-ttl').value = d.service.image_ttl_hours_effective || d.service.image_ttl_hours;
      setStatus(`并行 ${d.service.inflight_tasks}/${d.service.max_concurrent_tasks} · 24h ${d.requests.last_24h_ok}/${d.requests.last_24h} 成功 · 重试 ${d.requests.retried} 次 · 存储 ${fmtBytes(d.storage.bytes)} · PID ${d.service.pid}`);
      // 全局设置标签页字段
      const s = d.settings || {};
      document.getElementById('s-score').value = s.score_threshold ?? 0.52;
      document.getElementById('s-expand').value = s.expand_ratio ?? 0.30;
      document.getElementById('s-minface').value = s.min_face_skip ?? 50;
      document.getElementById('s-dot').value = s.dot_radius ?? 3;
      document.getElementById('s-step').value = s.face_grid_step ?? 14;
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
        ? `<img id="${id}" src="${escapeHtml(url)}" alt="${escapeHtml(label)}" loading="lazy" onload="showImgSize(this)" onclick="openImagePreview(this.src, this.alt)" data-size="${sizeBytes || 0}" />`
        : `<div class="image-empty" id="${id}">无可预览图片</div>`}</div>`;
    }
    function openImagePreview(url, label){
      const modal = document.getElementById('image-modal');
      const image = document.getElementById('image-modal-content');
      image.src = url;
      image.alt = label || '图片大图预览';
      modal.hidden = false;

      let scale = 1, tx = 0, ty = 0;
      let dragging = false, downX = 0, downY = 0, startTX = 0, startTY = 0;

      function clamp(){
        const w = (image.naturalWidth || image.width) * scale;
        const h = (image.naturalHeight || image.height) * scale;
        const mw = modal.clientWidth, mh = modal.clientHeight;
        const maxTX = Math.max(0, (w - mw) / 2 / scale);
        const maxTY = Math.max(0, (h - mh) / 2 / scale);
        tx = Math.min(maxTX, Math.max(-maxTX, tx));
        ty = Math.min(maxTY, Math.max(-maxTY, ty));
      }
      function apply(){
        image.style.transform = `scale(${scale}) translate(${tx}px, ${ty}px)`;
        image.style.transition = dragging ? 'none' : 'transform .15s ease';
        image.style.cursor = scale > 1 ? (dragging ? 'grabbing' : 'grab') : 'default';
      }

      image.onload = function(){ scale=1; tx=ty=0; apply(); };
      scale=1; tx=ty=0; apply();

      modal._zoomHandler = function(e){
        e.preventDefault();
        scale += e.deltaY < 0 ? 0.1 : -0.1;
        scale = Math.min(5, Math.max(0.3, scale));
        clamp(); apply();
      };
      modal.addEventListener('wheel', modal._zoomHandler, {passive:false});

      modal._dragStart = function(e){
        if(scale <= 1) return;
        dragging = true;
        downX = (e.touches ? e.touches[0].clientX : e.clientX);
        downY = (e.touches ? e.touches[0].clientY : e.clientY);
        startTX = tx; startTY = ty;
        apply(); e.preventDefault();
      };
      modal._dragMove = function(e){
        if(!dragging) return;
        const cx = e.touches ? e.touches[0].clientX : e.clientX;
        const cy = e.touches ? e.touches[0].clientY : e.clientY;
        tx = startTX + (cx - downX) / scale;
        ty = startTY + (cy - downY) / scale;
        clamp(); apply();
      };
      modal._dragEnd = function(){
        dragging = false; apply();
      };
      image.addEventListener('mousedown', modal._dragStart);
      image.addEventListener('touchstart', modal._dragStart, {passive:false});
      window.addEventListener('mousemove', modal._dragMove);
      window.addEventListener('touchmove', modal._dragMove, {passive:false});
      window.addEventListener('mouseup', modal._dragEnd);
      window.addEventListener('touchend', modal._dragEnd);
    }
    function closeImagePreview(event){
      if(event && event.target !== event.currentTarget) return;
      const modal = document.getElementById('image-modal');
      modal.hidden = true;
      const image = document.getElementById('image-modal-content');
      modal.removeEventListener('wheel', modal._zoomHandler);
      image.removeEventListener('mousedown', modal._dragStart);
      image.removeEventListener('touchstart', modal._dragStart);
      window.removeEventListener('mousemove', modal._dragMove);
      window.removeEventListener('touchmove', modal._dragMove);
      window.removeEventListener('mouseup', modal._dragEnd);
      window.removeEventListener('touchend', modal._dragEnd);
      delete modal._zoomHandler;
      delete modal._dragStart; delete modal._dragMove; delete modal._dragEnd;
      image.src = '';
    }
    async function loadRequests(){
      let url = '/api/admin/requests?offset=' + ((requestPage - 1) * requestPageSize) + '&limit=' + requestPageSize;
      if(activeParentSearch) url += '&parent_task_id=' + encodeURIComponent(activeParentSearch);
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
      let fUrl = '/api/admin/files?offset=0&limit=10';
      if(activeParentSearch) fUrl += '&parent_task_id=' + encodeURIComponent(activeParentSearch);
      const d = await api(fUrl);
      document.getElementById('files').innerHTML = d.items.map(x => {
        const tid = x.task_id || '';
        const onclick = tid ? `onclick="showTaskDetail('${escapeHtml(tid)}')"` : '';
        const style = tid ? 'style="cursor:pointer"' : '';
        const title = tid ? 'title="点击查看任务详情"' : '';
        return `<div class="file" ${onclick} ${style} ${title}><img src="${x.url}" loading="lazy" alt="${escapeHtml(x.name)}" onclick="event.stopPropagation();openImagePreview(this.src, this.alt)" /><div>${x.name}<br>${fmtBytes(x.size)}</div></div>`;
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
        return true;
      } catch(e) {
        setStatus(`任务查询失败: ${e.message}`);
        return false;
      }
    }
    async function findTask(){
      const q = document.getElementById('task-search').value.trim();
      const isTaskId = /^[0-9a-f]{32}$/i.test(q);
      if(isTaskId && await showTaskDetail(q)){
        activeParentSearch = '';
        return;
      }
      activeParentSearch = q;
      requestPage = 1;
      galleryPage = 1;
      await Promise.all([loadRequests(), loadFiles()]);
      if(activeTab === 'gallery') loadGalleryPage();
    }
    async function copyTaskId(){
      const taskId = document.getElementById('task-search').value.trim();
      if(!taskId) return;
      await navigator.clipboard.writeText(taskId);
      setStatus('任务 ID 已复制');
    }
    function showTab(name){
      activeTab = name;
      document.querySelectorAll('.tab-view').forEach(el => {
        const visible = el.dataset.tab === name || (name === 'overview' && el.dataset.tab === 'overview-stats');
        el.hidden = !visible;
      });
      document.querySelectorAll('.tab').forEach(el => el.classList.toggle('active', el.dataset.tab === name));
    }
    async function showGalleryTab(){
      showTab('gallery');
      await loadGalleryPage();
    }
    function showLabTab(){
      const t = tokenEl.value.trim();
      document.getElementById("lab-frame").src = "/lab?token=" + encodeURIComponent(t);
      showTab("lab");
    }
    async function loadGalleryPage(){
      let url = `/api/admin/files?offset=${(galleryPage - 1) * galleryPageSize}&limit=${galleryPageSize}`;
      if(activeParentSearch) url += '&parent_task_id=' + encodeURIComponent(activeParentSearch);
      const d = await api(url);
      galleryPageCount = Math.max(1, Math.ceil(d.total / galleryPageSize));
      if(galleryPage > galleryPageCount){ galleryPage = galleryPageCount; return loadGalleryPage(); }
      document.getElementById('gallery-files').innerHTML = d.items.map(x => {
        const tid = x.task_id || '';
        const onclick = tid ? `onclick="showTaskDetail('${escapeHtml(tid)}')"` : '';
        const style = tid ? 'style="cursor:pointer"' : '';
        const title = tid ? 'title="点击查看任务详情"' : '';
        return `<div class="file" ${onclick} ${style} ${title}><img src="${x.url}" loading="lazy" alt="${escapeHtml(x.name)}" onclick="event.stopPropagation();openImagePreview(this.src, this.alt)" /><div>${x.name}<br>${fmtBytes(x.size)}</div></div>`;
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
    async function clearCache(){
      if(!confirm("确认清空所有打码缓存？")) return;
      const d = await api("/api/admin/clear-cache", {method:"POST"});
      setStatus("已清空 L1 " + (d.cleared_l1 || 0) + " 条、L2 " + (d.cleared_l2 || 0) + " 条缓存");
    }
    async function clearParentCache(){
      const parentTaskId = activeParentSearch;
      if(!parentTaskId){ setStatus('请先查询父任务 ID'); return; }
      if(!confirm(`确认清除父任务 ${parentTaskId} 的缓存？`)) return;
      try {
        const url = '/api/admin/clear-cache?parent_task_id=' + encodeURIComponent(parentTaskId);
        const d = await api(url, {method:'POST'});
        setStatus(`已清除父任务缓存 L1 ${d.cleared_l1 || 0} 条、L2 ${d.cleared_l2 || 0} 条`);
      } catch(e) { setStatus(`清除父任务缓存失败: ${e.message}`); }
    }
    async function saveSettings(){
      const body = {
        max_concurrent_tasks: Number(document.getElementById('s-concurrency').value),
        max_retries: Number(document.getElementById('s-retries').value),
        retry_backoff_seconds: Number(document.getElementById('s-backoff').value),
        image_ttl_hours: Number(document.getElementById('s-ttl').value),
        score_threshold: Number(document.getElementById('s-score').value),
        expand_ratio: Number(document.getElementById('s-expand').value),
        min_face_skip: Number(document.getElementById('s-minface').value),
        dot_radius: Number(document.getElementById('s-dot').value),
        face_grid_step: Number(document.getElementById('s-step').value),
      };
      try {
        await api('/api/admin/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
        await loadSummary();
        setStatus('设置已保存');
      } catch(e) { setStatus(`设置保存失败: ${e.message}`); }
    }
    async function cleanup(){
      if(!confirm('按当前 TTL 清理过期静态图片？')) return;
      const d = await api('/api/admin/cleanup', {method:'POST'});
      setStatus(`已删除 ${d.deleted_files} 个文件，释放 ${fmtBytes(d.freed_bytes)}`);
      await loadAll();
    }
    async function loadSettingsTab(){
      try {
        const d = await api('/api/admin/settings');
        const s = d.settings || {};
        document.getElementById('s-concurrency').value = s.max_concurrent_tasks ?? 64;
        document.getElementById('s-retries').value = s.max_retries ?? 3;
        document.getElementById('s-backoff').value = s.retry_backoff_seconds ?? 3;
        document.getElementById('s-ttl').value = s.image_ttl_hours ?? 72;
        document.getElementById('s-score').value = s.score_threshold ?? 0.52;
        document.getElementById('s-expand').value = s.expand_ratio ?? 0.30;
        document.getElementById('s-minface').value = s.min_face_skip ?? 50;
        document.getElementById('s-dot').value = s.dot_radius ?? 3;
        document.getElementById('s-step').value = s.face_grid_step ?? 14;
      } catch(e) { setStatus(`全局设置加载失败: ${e.message}`); }
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
    document.addEventListener('keydown', event => { if(event.key === 'Escape') closeImagePreview(); });
    loadAll().then(() => {
      const match = location.hash.match(/^#task=(.+)$/);
      if(match) showTaskDetail(decodeURIComponent(match[1]));
    });
    setInterval(() => { if(tokenEl.value.trim()) loadAll(); }, 30000);
    setInterval(() => { if(tokenEl.value.trim()) loadSummary(); }, 3000);
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
