"""
对抗强度量化 v2: 用 SSIM + 关键点大块遮挡分析
================================================
核心指标:
  - SSIM(结构相似度, vs 原图): 越低 = 视觉差异越大 = 越对抗
  - 黑色像素占比 (>50% 是 solid/anti_face 的强遮挡)
  - 红色像素占比 (landmark 强干扰)
  - 关键点位置 (landmark/cv 用 cv 检测)
  - Laplacian 局部方差(脸部区域内的纹理残留度, 越低 = 越对抗)

注意: 此量化是"打码前后视觉差异"代理指标, 与真实 face_recognition 模型打分相关但不 100% 对应。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MODES = ["gaussian", "pixelate", "solid", "landmark",
         "landmark_whole_face", "anti_face"]


def face_crop_stats(img: np.ndarray, ref_img: np.ndarray) -> dict:
    """对一张图, 计算对抗强度代理指标."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img

    # SSIM (vs 原图)
    ssim = cv2.compareSSIM(ref_gray, gray) if hasattr(cv2, 'compareSSIM') else None
    if ssim is None:
        # 新版 OpenCV 可能在 image_quality 模块
        try:
            from skimage.metrics import structural_similarity
            ssim = structural_similarity(ref_gray, gray)
        except ImportError:
            ssim = None

    b, g, r = cv2.split(img)
    ref_b, ref_g, ref_r = cv2.split(ref_img)
    red_pct = float(np.mean((r > 150) & (g < 100) & (b < 100)) * 100)
    black_pct = float(np.mean(gray < 30) * 100)

    # 脸区域局部纹理残留 (Laplacian 方差, vs 原图)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ref_lap = float(cv2.Laplacian(ref_gray, cv2.CV_64F).var())

    return {
        "ssim": round(ssim, 3) if ssim is not None else None,
        "red_pct": round(red_pct, 2),
        "black_pct": round(black_pct, 2),
        "lap": round(lap, 1),
        "ref_lap": round(ref_lap, 1),
        "lap_residual_pct": round(lap / max(ref_lap, 1) * 100, 1),
    }


def find_file(img_dir: Path, mode: str, engine: str) -> Path | None:
    candidates = [
        img_dir / mode / f"{engine}.jpg",
        img_dir / f"{mode}_{engine}.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    img_dir = ROOT / "复核测试" / "202608061049233114_rXEZe"
    orig = cv2.imread(str(img_dir / "original.jpg"))
    # 对齐尺寸: process_image 在生成时缩到 max_side=1920, 这里把原图也对齐
    # 用第一张输出图的尺寸作参考
    sample = None
    for mode in MODES:
        for engine in ("cv", "ort"):
            f = find_file(img_dir, mode, engine)
            if f:
                sample = cv2.imread(str(f))
                break
        if sample is not None:
            break
    if sample is not None and sample.shape != orig.shape:
        orig = cv2.resize(orig, (sample.shape[1], sample.shape[0]),
                          interpolation=cv2.INTER_AREA)

    rows = []
    for mode in MODES:
        for engine in ("cv", "ort"):
            f = find_file(img_dir, mode, engine)
            if not f:
                continue
            img = cv2.imread(str(f))
            stats = face_crop_stats(img, orig)
            stats["mode"] = mode
            stats["engine"] = engine
            rows.append(stats)

    # 排序: SSIM 升序 (越低越对抗) + 黑色占比降序
    # 综合: 用 (1 - ssim) * 100 作为对抗分数(0=完全相同, 100=完全不同)
    def score(r):
        ssim_part = (1 - r["ssim"]) * 100 if r["ssim"] is not None else 0
        return ssim_part
    sorted_r = sorted(rows, key=score, reverse=True)

    print(f"=== 对抗强度排名 (SSIM 越低 = 视觉差异越大 = 越对抗) ===")
    print(f"原图 vs 原图: SSIM=1.000 (无差异)\n")
    for i, r in enumerate(sorted_r, 1):
        ssim_str = f"{r['ssim']:.3f}" if r['ssim'] is not None else "N/A"
        print(f"{i:2d}. [{r['engine']:3s}] {r['mode']:20s}  "
              f"SSIM={ssim_str}  red={r['red_pct']:5.1f}%  black={r['black_pct']:5.1f}%  "
              f"对抗分={score(r):5.1f}")

    (HERE / "anti_face_strength.json").write_text(json.dumps({
        "image": "202608061049233114_rXEZe",
        "rankings": sorted_r,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()