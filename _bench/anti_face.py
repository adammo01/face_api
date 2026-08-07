"""
新增模式: anti_face 重点遮挡(对抗人脸识别)
=====================================
思路: 同时破坏官方人脸识别系统依赖的所有关键信号:
  1. landmark 几何 -> 在眼/鼻/嘴 5 关键点画大块黑色矩形(破坏局部几何)
  2. 皮肤纹理 -> 整脸高斯模糊(大 ksize=51, 破坏微观纹理)
  3. landmark 全局 -> 整脸密集红点网格(扰乱全局 landmark 分布)

与现有 5 模式的差异:
  - gaussian/pixelate: 只破坏纹理, landmark 和脸型保留 -> 易被识别
  - solid: 完全破坏 -> 视觉太暴力
  - landmark(只点 5 点): landmark 几何没破坏, 反而被识别
  - landmark_whole_face: 破坏 landmark 但不破坏纹理, 中等对抗
  - anti_face(新增): 同时破坏 landmark + 纹理 + 几何 -> 最强对抗
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import face_blur as fb
from ort_detector import OrtYuNetDetector


def _apply_anti_face(region: np.ndarray,
                    landmarks: dict | None = None,
                    occlusion_size: int = 60,
                    blur_ksize: int = 51,
                    region_box: tuple | None = None,
                    color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    anti_face 打码:
      1) 整脸高斯模糊(破坏皮肤纹理)
      2) 在 5 关键点画大块黑色矩形(破坏 landmark 几何)
      3) 整脸密集红点网格(扰乱全局 landmark)
    """
    out = region.copy()
    rh, rw = out.shape[:2]
    if rh == 0 or rw == 0:
        return out

    # 第 1 层: 整脸高斯模糊
    out = cv2.GaussianBlur(out, (blur_ksize | 1, blur_ksize | 1), 0)

    # 第 2 层: 关键点大块黑色矩形
    if landmarks and region_box is not None:
        rx, ry, _, _ = region_box
        for name in ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth"):
            lx, ly = landmarks.get(name, (None, None))
            if lx is None:
                continue
            cx, cy = int(lx - rx), int(ly - ry)
            half = occlusion_size // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(rw, cx + half)
            y2 = min(rh, cy + half)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, -1)

    # 第 3 层: 整脸密集红点 (与 landmark_whole_face 类似)
    face_grid_step = 12
    dot_radius = 3
    half = face_grid_step // 2
    for yy in range(half, rh, face_grid_step):
        for xx in range(half, rw, face_grid_step):
            cv2.circle(out, (xx, yy), dot_radius, (0, 0, 255), -1, cv2.LINE_AA)

    return out


def process_anti_face(img_bytes: bytes, score_threshold: float = 0.6,
                     expand_ratio: float = 0.30,
                     occlusion_size: int = 60,
                     blur_ksize: int = 51,
                     engine: str = "cv") -> dict:
    """用 face_blur.process_image 框架, 复用检测 + 打码, 但打码步骤替换成 anti_face."""
    import threading
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图片")
    h, w = img.shape[:2]

    # 大图先缩
    max_side = 1920
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # 检测: cv 或 ort
    if engine == "cv":
        det = fb.FaceDetector(score_threshold=score_threshold)
        faces = det.detect_multiscale(img)
    else:
        det = OrtYuNetDetector(str(fb.MODEL_PATH),
                                score_threshold=score_threshold)
        boxes = det.detect_multiscale(img)
        faces = [fb.FaceBox(x, y, ww, hh, s, "ort")
                 for (x, y, ww, hh, s) in boxes]

    # 打码: 对每个检测框跑 anti_face
    for f in faces:
        box = f.expand(w, h, ratio=expand_ratio)
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
            if engine == "cv":
                sub_faces = det.detect_with_landmarks(sub)
            else:
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
        region = _apply_anti_face(
            region, landmarks=landmarks,
            occlusion_size=occlusion_size, blur_ksize=blur_ksize,
            region_box=(bx, by, bw, bh),
        )
        img[by:by + bh, bx:bx + bw] = region

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {
        "image_bytes": buf.tobytes(),
        "face_count": len(faces),
        "engine": engine,
    }


if __name__ == "__main__":
    import time, json
    img_path = sys.argv[1] if len(sys.argv) > 1 else \
        str(ROOT / "复核测试" / "202608061049233114_rXEZe" / "original.jpg")
    if not Path(img_path).exists():
        # fallback: 重新下
        import urllib.request
        url = "https://cdn.xingyuemeng.com/video/img_38552/202608061049233114_rXEZe.jpg"
        img_bytes = urllib.request.urlopen(url).read()
    else:
        img_bytes = Path(img_path).read_bytes()

    out_dir = Path("/tmp/anti_face_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    for engine in ("cv", "ort"):
        t0 = time.perf_counter()
        r = process_anti_face(img_bytes, engine=engine)
        ms = (time.perf_counter() - t0) * 1000
        (out_dir / f"anti_face_{engine}.jpg").write_bytes(r["image_bytes"])
        print(f"anti_face/{engine}: faces={r['face_count']} {ms:.1f}ms -> {out_dir}/anti_face_{engine}.jpg")