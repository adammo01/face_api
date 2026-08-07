"""
新增模式: 打码 + 红点 组合 (用户指定方案)
=========================================
用户需求: "打码加红点" / "高斯+红点" / 全黑的不行

组合一: gaussian_dots   = 整脸高斯模糊 + 整脸密集红点网格
组合二: pixelate_dots   = 整脸像素化(马赛克) + 整脸密集红点网格

与 landmark_whole_face 的区别: landmark_whole_face 只有红点;
  gaussian_dots / pixelate_dots 在红点下先铺一层打码(高斯/马赛克),
  把皮肤纹理也破坏掉, 同时红点扰乱 landmark 几何 —— 对抗更强。

实现: 基于 face_blur.process_image, 复用检测, 打码层替换成 [打码 + 红点]。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import face_blur as fb
from ort_detector import OrtYuNetDetector


def _apply_blur_dots(region: np.ndarray,
                     blur_mode: str,          # "gaussian" | "pixelate"
                     landmarks: dict | None = None,
                     ksize: int = 51,          # gaussian 核
                     block_size: int = 15,     # pixelate 块
                     face_grid_step: int = 12, # 红点网格间距
                     dot_radius: int = 3,
                     region_box: tuple | None = None) -> np.ndarray:
    """打码(高斯/像素化) + 红点网格叠加."""
    out = region.copy()
    rh, rw = out.shape[:2]
    if rh == 0 or rw == 0:
        return out

    # 第 1 层: 整脸打码
    if blur_mode == "gaussian":
        out = cv2.GaussianBlur(out, (ksize | 1, ksize | 1), 0)
    else:  # pixelate
        bs = max(2, block_size)
        small_w = max(1, rw // bs)
        small_h = max(1, rh // bs)
        small = cv2.resize(out, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        out = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

    # 第 2 层: 整脸密集红点网格 (扰乱 landmark 几何)
    half = face_grid_step // 2
    for yy in range(half, rh, face_grid_step):
        for xx in range(half, rw, face_grid_step):
            cv2.circle(out, (xx, yy), dot_radius, (0, 0, 255), -1, cv2.LINE_AA)

    return out


def process_blur_dots(img_bytes: bytes,
                      blur_mode: str,          # "gaussian" | "pixelate"
                      score_threshold: float = 0.6,
                      expand_ratio: float = 0.30,
                      ksize: int = 51,
                      block_size: int = 15,
                      face_grid_step: int = 12,
                      dot_radius: int = 3,
                      engine: str = "cv") -> dict:
    """检测 + [打码 + 红点] 打码. engine: cv | ort."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图片")
    h, w = img.shape[:2]

    max_side = 1920
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # 检测
    if engine == "cv":
        det = fb.FaceDetector(score_threshold=score_threshold)
        faces = det.detect_multiscale(img)
    else:
        det = OrtYuNetDetector(str(fb.MODEL_PATH), score_threshold=score_threshold)
        boxes = det.detect_multiscale(img)
        faces = [fb.FaceBox(x, y, ww, hh, s, "ort") for (x, y, ww, hh, s) in boxes]

    # 打码
    for f in faces:
        box = f.expand(w, h, ratio=expand_ratio)
        bx, by, bw, bh = box.x, box.y, box.w, box.h
        if bw < 20 or bh < 20:
            continue
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w, bx + bw), min(h, by + bh)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        region = img[by:by + bh, bx:bx + bw]
        region = _apply_blur_dots(
            region, blur_mode=blur_mode,
            ksize=ksize, block_size=block_size,
            face_grid_step=face_grid_step, dot_radius=dot_radius,
        )
        img[by:by + bh, bx:bx + bw] = region

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {
        "image_bytes": buf.tobytes(),
        "face_count": len(faces),
        "engine": engine,
        "mode": f"{blur_mode}_dots",
    }


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else \
        str(ROOT / "复核测试" / "202608061049233114_rXEZe" / "original.jpg")
    img_bytes = Path(img_path).read_bytes()

    out_dir = Path("/tmp/blur_dots_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    for blur_mode in ("gaussian", "pixelate"):
        for engine in ("cv", "ort"):
            t0 = time.perf_counter()
            r = process_blur_dots(img_bytes, blur_mode=blur_mode, engine=engine)
            ms = (time.perf_counter() - t0) * 1000
            name = f"{blur_mode}_dots_{engine}.jpg"
            (out_dir / name).write_bytes(r["image_bytes"])
            print(f"{blur_mode}_dots/{engine}: faces={r['face_count']} {ms:.1f}ms -> {out_dir}/{name}")