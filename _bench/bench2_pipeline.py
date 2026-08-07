"""
参数化流水线基准: engine x max_side x blur_algo 组合对比
=======================================================
用法:
  python bench2_pipeline.py --engine cv   --maxside 1920 --algo gaussian
  python bench2_pipeline.py --engine ort  --maxside 1280 --algo boxblur
  python bench2_pipeline.py --all          # 跑全部组合
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import face_blur as fb
from _bench.ort_detector import OrtYuNetDetector

# ---------------------------------------------------------------------------
# 打码算法: 在 face_blur 已有实现基础上补充 box blur
# ---------------------------------------------------------------------------

def _apply_box_blur(region: np.ndarray, ksize: int = 31, passes: int = 2) -> np.ndarray:
    """滑动窗 box blur (2 次均值模糊, 视觉接近高斯, 像素读取少)."""
    k = max(3, ksize | 1)
    out = region
    for _ in range(passes):
        out = cv2.blur(out, (k, k))
    return out


# ---------------------------------------------------------------------------
# 参数化检测器
# ---------------------------------------------------------------------------

def build_detector(engine: str, score_threshold: float = 0.6):
    if engine == "cv":
        return fb.FaceDetector(score_threshold=score_threshold)
    return OrtYuNetDetector(str(fb.MODEL_PATH), score_threshold=score_threshold,
                            nms_threshold=0.3, top_k=50)


# ---------------------------------------------------------------------------
# 参数化打码 (landmark 模式复用 face_blur 的实现)
# ---------------------------------------------------------------------------

def apply_blur_to_faces(img: np.ndarray, faces, mode: str, algo: str,
                        blur_params: dict, expand_ratio: float = 0.30) -> int:
    """对检测框执行打码, 返回处理的面数."""
    if mode in ("landmark", "landmark_whole_face"):
        return _apply_landmark_mode(img, faces, mode, blur_params, expand_ratio)
    h, w = img.shape[:2]
    for f in faces:
        box = f.expand(w, h, ratio=expand_ratio)
        region = img[box.y:box.y + box.h, box.x:box.x + box.w]
        if algo == "boxblur":
            ksize = blur_params.get("ksize", 31)
            region = _apply_box_blur(region, ksize=ksize)
        else:
            fn = fb.BLUR_MODES[mode]
            region = fn(region, **blur_params)
        img[box.y:box.y + box.h, box.x:box.x + box.w] = region
    return len(faces)


def _apply_landmark_mode(img: np.ndarray, faces, mode: str, blur_params: dict,
                         expand_ratio: float = 0.30) -> int:
    h, w = img.shape[:2]
    count = 0
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
        # 关键点: 尽量从 face 里取 (若检测器返回了 landmarks), 否则子区域检测
        landmarks = getattr(face, "landmarks", None)
        if landmarks is None and hasattr(detector_for, "detect_with_landmarks"):
            try:
                sub_faces = detector_for.detect_with_landmarks(sub)
                if sub_faces:
                    f0 = sub_faces[0]
                    lm = f0["landmarks"]
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
                region, landmarks=landmarks,
                face_grid_step=blur_params.get("face_grid_step", 14),
                dot_radius=blur_params.get("dot_radius", 3),
                grid_n=blur_params.get("grid_n", 5),
                spacing=blur_params.get("spacing", 14),
                color=blur_params.get("color", (0, 0, 255)),
                region_box=(bx, by, bw, bh),
            )
        else:
            region = fb._apply_landmark_dots(
                region, landmarks=landmarks,
                dot_radius=blur_params.get("dot_radius", 3),
                spacing=blur_params.get("spacing", 16),
                color=blur_params.get("color", (0, 0, 255)),
                region_box=(bx, by, bw, bh),
            )
        img[by:by + bh, bx:bx + bw] = region
        count += 1
    return count


# 全局引用, 供 landmark 模式取关键点
detector_for = None


def run_pipeline(img_bytes: bytes, engine: str, mode: str, max_side: int,
                 algo: str = "gaussian", n: int = 5) -> dict:
    """完整流水线: 解码 -> 降采样 -> 检测 -> 打码 -> 编码."""
    global detector_for
    detector = build_detector(engine)
    detector_for = detector

    times = []
    face_counts = []
    output_sizes = []
    for _ in range(n):
        t0 = time.perf_counter()
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        blur_params = {"dot_radius": 3, "spacing": 14,
                       "face_grid_step": 14, "grid_n": 5}
        expand_ratio = 0.30
        # 普通模式只传自身参数
        if mode == "gaussian":
            blur_params = {"ksize": 31}
        elif mode == "pixelate":
            blur_params = {"block_size": 15}
        elif mode == "solid":
            blur_params = {"color": (0, 0, 0)}

        # 检测: landmark 模式用多尺度, 其他用普通 detect_multiscale
        if mode in ("landmark", "landmark_whole_face"):
            if engine == "cv":
                faces_list = detector.detect_multiscale(img)
            else:
                boxes = detector.detect_multiscale(img)
                faces_list = [fb.FaceBox(x, y, ww, hh, s, "ort")
                              for (x, y, ww, hh, s) in boxes]
            n_faces = apply_blur_to_faces(img, faces_list, mode, algo, blur_params,
                                          expand_ratio)
        else:
            if engine == "cv":
                faces = detector.detect_multiscale(img)
            else:
                boxes = detector.detect_multiscale(img)
                faces = [fb.FaceBox(x, y, ww, hh, s, "ort")
                         for (x, y, ww, hh, s) in boxes]
            n_faces = apply_blur_to_faces(img, faces, mode, algo, blur_params,
                                          expand_ratio)

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        times.append((time.perf_counter() - t0) * 1000)
        face_counts.append(n_faces)
        output_sizes.append(len(buf.tobytes()) if ok else 0)

    return {
        "engine": engine, "mode": mode, "max_side": max_side, "algo": algo,
        "n": n,
        "times_ms": [round(t, 2) for t in times],
        "avg_ms": round(sum(times) / len(times), 2),
        "min_ms": round(min(times), 2),
        "p50_ms": round(sorted(times)[len(times) // 2], 2),
        "face_count": face_counts[0],
        "output_bytes": output_sizes[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=str(ROOT / "示例图片.jpg"))
    ap.add_argument("--engine", choices=["cv", "ort"], default="cv")
    ap.add_argument("--maxside", type=int, default=1920)
    ap.add_argument("--algo", choices=["gaussian", "pixelate", "solid", "boxblur"],
                    default="gaussian")
    ap.add_argument("--mode", default="gaussian", help="打码模式: gaussian/pixelate/solid/landmark/landmark_whole_face")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--all", action="store_true", help="跑全部组合")
    ap.add_argument("--out", default=str(HERE / "bench2_result.json"))
    args = ap.parse_args()

    img_bytes = (Path(args.img)).read_bytes()
    results = []

    if args.all:
        combos = []
        for engine in ("cv", "ort"):
            for mode in ("gaussian", "pixelate", "solid", "landmark_whole_face"):
                combos.append((engine, mode, 1920, "gaussian"))
        # 算法优化对比 (cv 引擎, gaussian 模式)
        for algo in ("gaussian", "boxblur"):
            combos.append(("cv", "gaussian", 1920, algo))
        for max_side in (1920, 1600, 1280):
            combos.append(("cv", "gaussian", max_side, "gaussian"))
    else:
        combos = [(args.engine, args.mode, args.maxside, args.algo)]

    for (engine, mode, max_side, algo) in combos:
        r = run_pipeline(img_bytes, engine, mode, max_side, algo, n=args.n)
        results.append(r)
        print(f"[{engine:3s}][{mode:20s}][maxside={max_side}][algo={algo:9s}] "
              f"avg={r['avg_ms']:>7.1f}ms p50={r['p50_ms']:>7.1f}ms "
              f"faces={r['face_count']} out={r['output_bytes']}B")

    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
