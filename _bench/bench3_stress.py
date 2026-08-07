"""
压力测试脚本: 并发请求 /api/face_blur, 统计 QPS / P95 / 失败率
==============================================================
用法:
  python bench3_stress.py --url http://127.0.0.1:8000 \
      --img-url http://127.0.0.1:18999/示例图片.jpg \
      --mode gaussian --concurrency 8 --total 200
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests

HERE = Path(__file__).resolve().parent


def worker(task_id: int, base: str, img_url: str, mode: str, results: list):
    # 加随机 query 避免命中 L1/L2 缓存, 测真实处理路径
    sep = "&" if "?" in img_url else "?"
    payload = {
        "image_url": f"{img_url}{sep}_nc={task_id}",
        "mode": mode,
        "score_threshold": 0.6,
        "expand_ratio": 0.30,
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{base}/api/face_blur", json=payload, timeout=60)
        dt = (time.perf_counter() - t0) * 1000
        ok = r.status_code == 200
        try:
            body = r.json()
        except Exception:
            body = {}
        results.append({
            "task_id": task_id, "http": r.status_code, "ok": ok,
            "latency_ms": round(dt, 2),
            "face_count": body.get("face_count"),
            "cached": body.get("cached", False),
            "error": body.get("error"),
        })
    except Exception as e:  # noqa: BLE001
        dt = (time.perf_counter() - t0) * 1000
        results.append({
            "task_id": task_id, "http": 0, "ok": False,
            "latency_ms": round(dt, 2), "error": str(e)[:200],
        })


def summarize(results: list, label: str, total_seconds: float):
    lats = sorted(r["latency_ms"] for r in results)
    ok = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    n = len(results)
    qps = n / total_seconds if total_seconds > 0 else 0
    def pct(p):
        if not lats:
            return 0
        return lats[min(len(lats) - 1, int(len(lats) * p))]

    return {
        "label": label,
        "total": n,
        "ok": len(ok),
        "failed": len(fails),
        "fail_rate": round(len(fails) / n * 100, 2) if n else 0,
        "qps": round(qps, 2),
        "total_seconds": round(total_seconds, 2),
        "latency_ms": {
            "min": round(lats[0], 2) if lats else 0,
            "p50": round(pct(0.5), 2),
            "p90": round(pct(0.9), 2),
            "p95": round(pct(0.95), 2),
            "p99": round(pct(0.99), 2),
            "max": round(lats[-1], 2) if lats else 0,
        },
        "face_count_avg": round(sum(r["face_count"] or 0 for r in ok) / len(ok), 2) if ok else 0,
        "cache_hits": sum(1 for r in ok if r["cached"]),
        "errors_sample": [r["error"] for r in fails[:5]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--img-url", default="http://127.0.0.1:18999/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87.jpg")
    ap.add_argument("--mode", default="gaussian")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--total", type=int, default=100)
    ap.add_argument("--out", default=str(HERE / "bench3_result.json"))
    args = ap.parse_args()

    # 预热 1 个请求 (触发模型加载)
    try:
        requests.post(f"{args.url}/api/face_blur",
                      json={"image_url": args.img_url, "mode": args.mode,
                            "score_threshold": 0.6, "expand_ratio": 0.30},
                      timeout=120)
        print("[warmup] done")
    except Exception as e:  # noqa: BLE001
        print(f"[warmup] failed: {e}")

    results = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, i, args.url, args.img_url, args.mode, results)
                for i in range(args.total)]
        for f in futs:
            f.result()
    total_seconds = time.perf_counter() - t_start

    summary = summarize(results, f"{args.mode}@c{args.concurrency}", total_seconds)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 追加到结果文件
    out_path = Path(args.out)
    existing = []
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    existing.append(summary)
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
