"""
差异图深度分析: 对比 cv vs ort 在 3 张不一致图上的检出框/score
也验证: 降低 score_threshold 后 ort 能否找回 cv 检出的脸 (阈值边缘 vs 真漏检)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import face_blur as fb
from ort_detector import OrtYuNetDetector

DIFF_IDX = [0, 1, 8]  # 检出数不一致的图
MANIFEST = json.loads((HERE / "review_images" / "manifest.json").read_text())


def analyze(idx: int):
    item = MANIFEST[idx]
    img = cv2.imread(str(HERE / "review_images" / item["file"]))
    h, w = img.shape[:2]
    # 服务端先缩到 1920
    if max(h, w) > 1920:
        s = 1920 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    print(f"\n===== [{idx}] {item['file']} ({w}x{h}) =====")

    cv_det = fb.FaceDetector(score_threshold=0.6)
    cv_faces = cv_det.detect_multiscale(img)
    print(f"cv  (th=0.6): {len(cv_faces)} 张")
    for f in sorted(cv_faces, key=lambda x: -x.score):
        print(f"    box=({f.x},{f.y},{f.w},{f.h}) score={f.score:.3f} src={f.source}")

    ort_det = OrtYuNetDetector(str(fb.MODEL_PATH), score_threshold=0.6)
    ort_faces = ort_det.detect_multiscale(img)
    print(f"ort (th=0.6): {len(ort_faces)} 张")
    for (x, y, bw, bh, s) in sorted(ort_faces, key=lambda x: -x[4]):
        print(f"    box=({x},{y},{bw},{bh}) score={s:.3f}")

    # ort 降低阈值到 0.3, 看能否找回 cv 检出的
    ort_det2 = OrtYuNetDetector(str(fb.MODEL_PATH), score_threshold=0.3)
    ort_faces2 = ort_det2.detect_multiscale(img)
    print(f"ort (th=0.3): {len(ort_faces2)} 张")
    for (x, y, bw, bh, s) in sorted(ort_faces2, key=lambda x: -x[4]):
        print(f"    box=({x},{y},{bw},{bh}) score={s:.3f}")


if __name__ == "__main__":
    for i in DIFF_IDX:
        analyze(i)
