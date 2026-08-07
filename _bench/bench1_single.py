"""
基准测试脚本集 (本地, 不改生产代码)
==================================
用法:
  python bench1_single.py                # 单图 5 模式基准 (OpenCV DNN 现状)
  python bench1_single.py --engine ort   # 换成 onnxruntime 推理
  python bench1_single.py --maxside 1280 # 输入降采样对比
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

MODES = ["pixelate", "gaussian", "solid", "landmark", "landmark_whole_face"]


def load_image(path: Path) -> bytes:
    return path.read_bytes()


def bench_process_image(img_bytes: bytes, mode: str, engine: str, max_side: int,
                        n: int = 5) -> dict:
    """对 process_image 做 n 次计时, 返回平均/最快/最慢."""
    if engine == "ort":
        import face_blur as fb
        from _bench.ort_detector import OrtYuNetDetector
        # 临时把 face_blur 的检测器换成 ORT 版 (进程内 patch)
        original = fb.FaceDetector

        class ORTDetectorWrapper(original):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                model_path = fb.MODEL_PATH
                self._ort = OrtYuNetDetector(
                    str(model_path),
                    score_threshold=kw.get("score_threshold", 0.6),
                    nms_threshold=kw.get("nms_threshold", 0.3),
                    top_k=kw.get("top_k", 50),
                )

            def detect(self, img_bgr):
                return [fb.FaceBox(x, y, w, h, s, "ort")
                        for (x, y, w, h, s) in self._ort.detect(img_bgr)]

            def detect_with_landmarks(self, img_bgr):
                h, w = img_bgr.shape[:2]
                out = []
                for (x, y, bw, bh, s) in self._ort.detect(img_bgr):
                    out.append({
                        "x": x, "y": y, "w": bw, "h": bh, "score": s,
                        "landmarks": _approx_landmarks(img_bgr, x, y, bw, bh),
                    })
                return out

        def _approx_landmarks(img_bgr, x, y, w, h):
            """ORT 版 landmark 简化: 用几何近似 (眼/鼻/嘴), 仅供耗时基准."""
            cx, cy = x + w / 2, y + h / 2
            return {
                "right_eye": (cx - w * 0.25, cy - h * 0.18),
                "left_eye": (cx + w * 0.25, cy - h * 0.18),
                "nose": (cx, cy + h * 0.05),
                "right_mouth": (cx - w * 0.15, cy + h * 0.35),
                "left_mouth": (cx + w * 0.15, cy + h * 0.35),
            }

        fb.FaceDetector = ORTDetectorWrapper
        # 清掉线程局部 detector, 强制重建
        fb._DETECTOR_LOCAL = __import__("threading").local()

    import face_blur as fb

    # 注入 max_side (临时改常量)
    original_max_side = fb.process_image.__globals__.get("_max_side", 1920)
    # process_image 里 _max_side 是局部变量, 需要 patch 源码级逻辑 -> 用 monkeypatch 函数
    # 更简单: 直接改全局? 不行, 是函数内局部。改用包装:
    times = []
    results = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fb.process_image(
            img_bytes, mode=mode,
            score_threshold=0.6, expand_ratio=0.30, return_faces=True,
        )
        times.append((time.perf_counter() - t0) * 1000)
        results.append(r)
    return {
        "mode": mode,
        "engine": engine,
        "n": n,
        "times_ms": times,
        "avg_ms": round(sum(times) / len(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "face_count": results[0]["face_count"],
        "output_bytes": len(results[0]["image_bytes"]),
        "output_path": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=str(ROOT / "示例图片.jpg"))
    ap.add_argument("--engine", choices=["cv", "ort"], default="cv")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=str(HERE / "bench1_result.json"))
    args = ap.parse_args()

    img_bytes = load_image(Path(args.img))
    modes = args.modes.split(",")
    all_results = []
    for m in modes:
        r = bench_process_image(img_bytes, m, args.engine, 1920, n=args.n)
        all_results.append(r)
        print(f"[{m:20s}] {args.engine:3s} avg={r['avg_ms']:>8.1f}ms "
              f"min={r['min_ms']:>8.1f}ms max={r['max_ms']:>8.1f}ms "
              f"faces={r['face_count']} out={r['output_bytes']}B")

    with open(args.out, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
