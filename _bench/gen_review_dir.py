"""
复核测试: 9 张真实业务图 × 5 种打码模式 × 2 种引擎 (cv / ort)
输出结构:
  复核测试/
    <图片名>/
      gaussian/
        cv.jpg
        ort.jpg
      pixelate/
        cv.jpg
        ort.jpg
      solid/
        cv.jpg
        ort.jpg
      landmark/
        cv.jpg
        ort.jpg
      landmark_whole_face/
        cv.jpg
        ort.jpg
   复核测试报告.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import face_blur as fb
from ort_detector import OrtYuNetDetector

MANIFEST = json.loads((HERE / "review_images" / "manifest.json").read_text())
MODES = ["gaussian", "pixelate", "solid", "landmark", "landmark_whole_face"]
OUT_ROOT = HERE.parent / "复核测试"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 各模式默认参数 (与 API 默认一致)
MODE_PARAMS = {
    "gaussian": {"ksize": 31},
    "pixelate": {"block_size": 15},
    "solid": {"color": (0, 0, 0)},
    "landmark": {"dot_radius": 3, "spacing": 16},
    "landmark_whole_face": {"dot_radius": 3, "spacing": 14,
                            "face_grid_step": 14, "grid_n": 5},
}


def process_cv(img_bytes: bytes, mode: str) -> dict:
    t0 = time.perf_counter()
    r = fb.process_image(img_bytes, mode=mode, score_threshold=0.6,
                         expand_ratio=0.30, return_faces=True,
                         **MODE_PARAMS[mode])
    r["engine_ms"] = (time.perf_counter() - t0) * 1000
    return r


def process_ort(img_bytes: bytes, mode: str) -> dict:
    """ORT 版: monkeypatch FaceDetector 后调用同一 process_image."""
    det = OrtYuNetDetector(str(fb.MODEL_PATH), score_threshold=0.6,
                           nms_threshold=0.3, top_k=50)

    # 包装成 face_blur.FaceDetector 兼容接口
    class ORTDet:
        def __init__(self, score_threshold=0.6):
            self._det = det
            self._th = score_threshold

        def getScoreThreshold(self):
            return self._th

        def detect(self, img_bgr):
            return [fb.FaceBox(x, y, ww, hh, s, "ort")
                    for (x, y, ww, hh, s) in self._det.detect(img_bgr)]

        def detect_with_landmarks(self, img_bgr):
            out = []
            for d in self._det.detect_with_landmarks(img_bgr):
                out.append({"x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"],
                            "score": d["score"],
                            "landmarks": {k: tuple(v) for k, v in d["landmarks"].items()}})
            return out

        def detect_multiscale(self, img_bgr, target_long_sides=(640, 1600),
                              use_haar_fallback=False):
            boxes = self._det.detect_multiscale(img_bgr, target_long_sides)
            return [fb.FaceBox(x, y, ww, hh, s, "ort")
                    for (x, y, ww, hh, s) in boxes]

        def detect_haar_fallback(self, img_bgr):
            return []

    old_cls = fb.FaceDetector
    fb.FaceDetector = ORTDet
    fb._DETECTOR_LOCAL = __import__("threading").local()
    try:
        t0 = time.perf_counter()
        r = fb.process_image(img_bytes, mode=mode, score_threshold=0.6,
                             expand_ratio=0.30, return_faces=True,
                             **MODE_PARAMS[mode])
        r["engine_ms"] = (time.perf_counter() - t0) * 1000
        return r
    finally:
        fb.FaceDetector = old_cls
        fb._DETECTOR_LOCAL = __import__("threading").local()


def main():
    summary_rows = []
    for item in MANIFEST:
        idx = item["idx"]
        src_name = Path(item["file"]).name
        # 图片文件夹: 用可读名称 (去掉 review_XX_ 前缀, 保留原文件名)
        pretty = src_name.split("_", 2)[-1]
        img_dir = OUT_ROOT / pretty.rsplit(".", 1)[0]
        img_dir.mkdir(parents=True, exist_ok=True)

        img_bytes = (HERE / "review_images" / item["file"]).read_bytes()
        print(f"\n===== [{idx}] {src_name} ({item['width']}x{item['height']}) =====")

        row = {"idx": idx, "file": pretty, "size": f"{item['width']}x{item['height']}"}
        for mode in MODES:
            mode_dir = img_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)

            r_cv = process_cv(img_bytes, mode)
            (mode_dir / "cv.jpg").write_bytes(r_cv["image_bytes"])
            r_ort = process_ort(img_bytes, mode)
            (mode_dir / "ort.jpg").write_bytes(r_ort["image_bytes"])

            # 像素差异率 (同一张图两种引擎输出)
            a = cv2.imdecode(np.frombuffer(r_cv["image_bytes"], np.uint8), cv2.IMREAD_COLOR)
            b = cv2.imdecode(np.frombuffer(r_ort["image_bytes"], np.uint8), cv2.IMREAD_COLOR)
            diff = cv2.absdiff(a, b)
            same_pct = float(np.mean(diff < 10) * 100)
            mean_diff = float(diff.mean())

            row[f"{mode}_faces"] = f"{r_cv['face_count']}/{r_ort['face_count']}"
            row[f"{mode}_ms"] = f"{r_cv['engine_ms']:.1f}/{r_ort['engine_ms']:.1f}"
            row[f"{mode}_same"] = same_pct
            row[f"{mode}_mean"] = mean_diff
            print(f"  [{mode:20s}] faces={r_cv['face_count']}/{r_ort['face_count']} "
                  f"ms={r_cv['engine_ms']:.1f}/{r_ort['engine_ms']:.1f} "
                  f"一致率={same_pct:.1f}%")

        # 保存原图副本 (方便对照)
        cv2.imwrite(str(img_dir / "original.jpg"),
                    cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        summary_rows.append(row)

    (HERE / "review_compare.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"\n全部输出 -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
