"""
9 张真实业务图复核: cv (8000) vs ort (8001) 速度对比
- 每张图每种方案每种模式测 3 次, 取中位数 (process_ms = 纯检测+打码, elapsed_ms = 端到端含下载)
- URL 加随机 query 规避 L1/L2 缓存
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from urllib.parse import quote

import requests

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "review_images" / "manifest.json").read_text())

CV = "http://127.0.0.1:8000"
ORT = "http://127.0.0.1:8001"
IMG_BASE = "http://127.0.0.1:18999/_bench/review_images/"
MODES = ["gaussian", "landmark_whole_face"]
REPEATS = 3


def call(base: str, img_path: str, mode: str, run: int) -> dict:
    url = f"{IMG_BASE}{quote(img_path)}?_nc={int(time.time()*1000)}_{run}"
    t0 = time.perf_counter()
    r = requests.post(f"{base}/api/face_blur",
                      json={"image_url": url, "mode": mode,
                            "score_threshold": 0.6, "expand_ratio": 0.30},
                      timeout=180)
    dt = (time.perf_counter() - t0) * 1000
    d = r.json()
    return {
        "http": r.status_code,
        "face_count": d.get("face_count"),
        "blocked": d.get("blocked"),
        "process_ms": d.get("process_ms"),
        "elapsed_ms": d.get("elapsed_ms"),
        "client_total_ms": round(dt, 1),
        "error": d.get("error"),
    }


def median_key(results, key):
    vals = [r[key] for r in results if r.get(key) is not None]
    return round(statistics.median(vals), 1) if vals else None


def main():
    rows = []
    for item in MANIFEST:
        idx = item["idx"]
        fname = item["file"]
        w, h = item.get("width"), item.get("height")
        row = {"idx": idx, "file": fname, "size": f"{w}x{h}", "bytes_kb": round(item["bytes"]/1024)}
        for mode in MODES:
            for engine, base in (("cv", CV), ("ort", ORT)):
                # 首次请求触发模型/图缓存, 不纳入; 正式跑 REPEATS 次
                call(base, fname, mode, run=999)
                results = [call(base, fname, mode, run=i) for i in range(REPEATS)]
                row[f"{mode}_{engine}_faces"] = results[0].get("face_count")
                row[f"{mode}_{engine}_proc"] = median_key(results, "process_ms")
                row[f"{mode}_{engine}_e2e"] = median_key(results, "elapsed_ms")
                row[f"{mode}_{engine}_err"] = any(r.get("error") for r in results)
        rows.append(row)
        print(f"[{idx}] {fname[:40]}... faces(cv/ort)="
              f"{row.get('gaussian_cv_faces')}/{row.get('gaussian_ort_faces')} "
              f"proc={row.get('gaussian_cv_proc')}/{row.get('gaussian_ort_proc')}ms")
    out = HERE / "review_result.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
