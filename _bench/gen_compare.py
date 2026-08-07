"""
生成打码效果对比图: 原图 / cv-gaussian / ort-gaussian / pixelate / boxblur / landmark_whole_face
用于报告可视化与效果一致性验证
"""
from __future__ import annotations

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


def build_detector(engine):
    if engine == "cv":
        return fb.FaceDetector(score_threshold=0.6)
    return OrtYuNetDetector(str(fb.MODEL_PATH), score_threshold=0.6)


def detect_and_blur(img, engine, mode, max_side=1920):
    """cv/ort 检测 + 指定打码, 返回结果图"""
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    det = build_detector(engine)

    if mode in ("landmark", "landmark_whole_face"):
        if engine == "cv":
            faces = det.detect_multiscale(img)
        else:
            boxes = det.detect_multiscale(img)
            faces = [fb.FaceBox(x, y, ww, hh, s, "ort") for (x, y, ww, hh, s) in boxes]
        expand_ratio = 0.30
        for face in faces:
            box = face.expand(w, h, ratio=expand_ratio)
            bx, by, bw, bh = box.x, box.y, box.w, box.h
            if bw < 20 or bh < 20:
                continue
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(w, bx + bw), min(h, by + bh)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            sub = img[y1:y2, x1:x2]
            landmarks = None
            try:
                sub_faces = det.detect_with_landmarks(sub)
                if sub_faces:
                    lm = sub_faces[0]["landmarks"]
                    landmarks = {
                        "right_eye": (lm["right_eye"][0] + x1, lm["right_eye"][1] + y1),
                        "left_eye": (lm["left_eye"][0] + x1, lm["left_eye"][1] + y1),
                        "nose": (lm["nose"][0] + x1, lm["nose"][1] + y1),
                        "right_mouth": (lm["right_mouth"][0] + x1, lm["right_mouth"][1] + y1),
                        "left_mouth": (lm["left_mouth"][0] + x1, lm["left_mouth"][1] + y1),
                    }
            except Exception:
                pass
            region = img[by:by + bh, bx:bx + bw]
            if mode == "landmark_whole_face":
                region = fb._apply_landmark_whole_face_with_landmarks(
                    region, landmarks=landmarks, face_grid_step=14,
                    dot_radius=3, grid_n=5, spacing=14, color=(0, 0, 255),
                    region_box=(bx, by, bw, bh))
            else:
                region = fb._apply_landmark_dots(
                    region, landmarks=landmarks, dot_radius=3, spacing=16,
                    color=(0, 0, 255), region_box=(bx, by, bw, bh))
            img[by:by + bh, bx:bx + bw] = region
    else:
        if engine == "cv":
            faces = det.detect_multiscale(img)
        else:
            boxes = det.detect_multiscale(img)
            faces = [fb.FaceBox(x, y, ww, hh, s, "ort") for (x, y, ww, hh, s) in boxes]
        for face in faces:
            box = face.expand(w, h, ratio=0.30)
            region = img[box.y:box.y + box.h, box.x:box.x + box.w]
            img[box.y:box.y + box.h, box.x:box.x + box.w] = fb.BLUR_MODES[mode](region)
    return img


def main():
    out_dir = HERE / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(ROOT / "示例图片.jpg"))
    print(f"原图: {src.shape}")

    # 效果对比: 先缩到 1280 长边便于展示 (仅展示用, 基准用 1920)
    h, w = src.shape[:2]
    scale = 1280 / max(h, w)
    small = cv2.resize(src, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(out_dir / "0_original.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 90])

    panels = [
        ("1_cv_gaussian.jpg",   "cv",  "gaussian"),
        ("2_ort_gaussian.jpg",  "ort", "gaussian"),
        ("3_cv_pixelate.jpg",   "cv",  "pixelate"),
        ("4_cv_landmark_whole.jpg", "cv", "landmark_whole_face"),
        ("5_ort_landmark_whole.jpg","ort","landmark_whole_face"),
    ]
    for fname, engine, mode in panels:
        img = detect_and_blur(small.copy(), engine, mode, max_side=1280)
        cv2.imwrite(str(out_dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {fname} OK ({engine}/{mode})")

    print(f"\n对比图输出 -> {out_dir}")


if __name__ == "__main__":
    main()
