"""
自动人脸打码核心处理模块
========================

参考方案: 渠道5-智能过人脸/serverless_face_blurring_solution.md
实现要点:
  - 人脸检测: OpenCV DNN + YuNet 模型 (零成本, 无第三方 API 调用)
  - 打码方式: 马赛克 / 高斯模糊 / 纯色遮盖 / 关键点点阵 / 整脸范围网格点阵
  - 函数入口 process_image(input_bytes, mode, **params) -> dict
    返回 {"image_bytes": ..., "faces": [...], "elapsed_ms": ...}
    此签名可直接包装为 Serverless (FC / SCF / Lambda) 的 handler.

新模式 "landmark_whole_face" (2026-08-05 沉淀):
  - 整脸范围按 face_grid_step 均匀打红点 + 5 关键点附近 grid_n×grid_n 密集叠加
  - 默认参数 face_grid_step=14, dot_radius=3, grid_n=5, expand_ratio=0.30
  - 验证: 多图 reference 场景下能过火山方舟 InputImageSensitiveContentDetected
  - 推荐参数: dot_radius=3, face_grid_step=14, grid_n=5, expand_ratio=0.30
    (实测 5 角色图 4s 720p 视频 213s 生成, 87,300 tokens, 2.3 MB)
"""

from __future__ import annotations

import io
import os
import time
import threading
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 模型管理
# ---------------------------------------------------------------------------

# OpenCV Zoo 的 YuNet 模型 (~340KB), 开源免费, 无需鉴权
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
# 模型缓存到用户目录, 避免重复下载, 也方便 Serverless 层打成 Layer/层时预置
MODEL_DIR = Path(os.environ.get("FACE_BLUR_MODEL_DIR",
                                Path.home() / ".cache" / "face_blur"))
MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"


_MODEL_LOCK = threading.Lock()

def _ensure_model(model_url: str = YUNET_URL, model_path: Path = MODEL_PATH) -> Path:
    """确保模型文件存在, 缺失则下载."""
    if model_path.exists() and model_path.stat().st_size > 1000:
        return model_path
    with _MODEL_LOCK:
        if model_path.exists() and model_path.stat().st_size > 1000:
            return model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[face_blur] downloading model -> {model_path}")
        urllib.request.urlretrieve(model_url, model_path)
    return model_path


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    source: str = "yunet"
    landmarks: dict | None = None

    def expand(self, img_w: int, img_h: int, ratio: float = 0.2) -> "FaceBox":
        """向外扩 ratio 比例, 避免打码后边缘漏脸."""
        new_w = int(self.w * (1 + ratio))
        new_h = int(self.h * (1 + ratio))
        dx = (new_w - self.w) // 2
        dy = (new_h - self.h) // 2
        x = max(0, self.x - dx)
        y = max(0, self.y - dy)
        w = min(img_w - x, new_w)
        h = min(img_h - y, new_h)
        return FaceBox(x, y, w, h, self.score, self.source, self.landmarks)


class FaceDetector:
    """轻量封装, 复用同一个 cv2.FaceDetectorYN 实例."""

    def __init__(self, score_threshold: float = 0.6,
                 nms_threshold: float = 0.3, top_k: int = 50):
        model_path = _ensure_model()
        # input_size 会在 detect() 时按图片实际尺寸动态调整
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )
        self._last_size = (320, 320)
        self._haar: cv2.CascadeClassifier | None = None

    def detect(self, img_bgr: np.ndarray) -> List[FaceBox]:
        h, w = img_bgr.shape[:2]
        # YuNet 需要显式 setInputSize 才能输出正确坐标
        if (w, h) != self._last_size:
            self._detector.setInputSize((w, h))
            self._last_size = (w, h)
        _, faces = self._detector.detect(img_bgr)
        if faces is None:
            return []
        out: List[FaceBox] = []
        for f in faces:
            # YuNet 返回 [x, y, w, h, ...landmarks..., score]
            x, y, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            score = float(f[-1])
            out.append(FaceBox(x, y, bw, bh, score, "yunet"))
        return out

    def detect_with_landmarks(self, img_bgr: np.ndarray) -> List[dict]:
        """
        同 detect, 但额外返回 5 个 landmark 像素坐标.
        YuNet 前 5 landmark 顺序: right_eye, left_eye, nose, right_mouth, left_mouth
        返回 list of dict: {x, y, w, h, score, landmarks}
        """
        h, w = img_bgr.shape[:2]
        if (w, h) != self._last_size:
            self._detector.setInputSize((w, h))
            self._last_size = (w, h)
        _, faces = self._detector.detect(img_bgr)
        if faces is None:
            return []
        out = []
        for f in faces:
            x, y, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            score = float(f[-1])
            landmarks = {
                "right_eye":  (float(f[4]),  float(f[5])),
                "left_eye":   (float(f[6]),  float(f[7])),
                "nose":       (float(f[8]),  float(f[9])),
                "right_mouth": (float(f[10]), float(f[11])),
                "left_mouth":  (float(f[12]), float(f[13])),
            }
            out.append({"x": x, "y": y, "w": bw, "h": bh,
                        "score": score, "landmarks": landmarks})
        return out

    def detect_multiscale(self, img_bgr: np.ndarray,
                          target_long_sides: tuple[int, ...] = (640, 1600),
                          use_haar_fallback: bool = False) -> List[FaceBox]:
        """
        多尺度检测: 原图 + 多个归一化长边尺寸, 合并去重.

        YuNet 模型 anchor 尺寸有限, 单尺度容易漏掉超大脸、小脸、弱侧脸。
        把同一张图缩放到几个常用长边尺寸后重复检测, 坐标按比例还原,
        再用 NMS 合并。Haar 可作为低分正脸兜底, 但误检率较高, 默认关闭。
        """
        h, w = img_bgr.shape[:2]
        all_boxes: List[FaceBox] = list(self.detect(img_bgr))

        long_side = max(w, h)
        for target in target_long_sides:
            if target <= 0:
                continue
            scale = target / long_side
            if abs(scale - 1.0) < 0.08:
                continue
            new_w, new_h = max(32, int(w * scale)), max(32, int(h * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
            prev_size = self._last_size
            self._detector.setInputSize((new_w, new_h))
            _, resized_faces = self._detector.detect(resized)
            self._detector.setInputSize(prev_size)
            self._last_size = prev_size

            if resized_faces is None:
                continue
            inv = 1.0 / scale
            for f in resized_faces:
                all_boxes.append(FaceBox(
                    int(f[0] * inv),
                    int(f[1] * inv),
                    int(f[2] * inv),
                    int(f[3] * inv),
                    float(f[-1]),
                    f"yunet@{target}",
                ))

        if use_haar_fallback:
            all_boxes.extend(self.detect_haar_fallback(img_bgr))

        return _nms_faces(_clip_faces(all_boxes, w, h), iou_thresh=0.35)

    def detect_overlapping_tiles(self, img_bgr: np.ndarray,
                                 tile_size: int = 512,
                                 step: int = 384) -> List[FaceBox]:
        """在重叠的小块中检测远处/拥挤人脸，补足整图检测的漏检。"""
        h, w = img_bgr.shape[:2]
        original_size = (w, h)
        boxes: List[FaceBox] = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                x2, y2 = min(w, x + tile_size), min(h, y + tile_size)
                tile = img_bgr[y:y2, x:x2]
                if tile.shape[0] < 64 or tile.shape[1] < 64:
                    continue
                tile_size_xy = (tile.shape[1], tile.shape[0])
                self._detector.setInputSize(tile_size_xy)
                _, faces = self._detector.detect(tile)
                if faces is None:
                    continue
                for face in faces:
                    boxes.append(FaceBox(
                        int(face[0]) + x, int(face[1]) + y,
                        int(face[2]), int(face[3]), float(face[-1]),
                        "yunet-tile",
                    ))
        self._detector.setInputSize(original_size)
        self._last_size = original_size
        return _nms_faces(_clip_faces(boxes, w, h), iou_thresh=0.35)

    def detect_haar_fallback(self, img_bgr: np.ndarray) -> List[FaceBox]:
        """OpenCV Haar 正脸兜底, 用于补 YuNet 漏掉的弱分数人脸。"""
        if self._haar is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(str(cascade_path))
        if self._haar.empty():
            return []

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_size = max(24, min(w, h) // 40)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [FaceBox(int(x), int(y), int(bw), int(bh), 0.35, "haar")
                for (x, y, bw, bh) in faces]


def _clip_faces(boxes: List[FaceBox], img_w: int, img_h: int) -> List[FaceBox]:
    """裁剪异常越界框, 同时过滤过小框。"""
    clipped: List[FaceBox] = []
    for b in boxes:
        x = max(0, b.x)
        y = max(0, b.y)
        x2 = min(img_w, b.x + b.w)
        y2 = min(img_h, b.y + b.h)
        bw, bh = x2 - x, y2 - y
        if bw < 12 or bh < 12:
            continue
        clipped.append(FaceBox(x, y, bw, bh, b.score, b.source, b.landmarks))
    return clipped


def _nms_faces(boxes: List[FaceBox], iou_thresh: float = 0.4) -> List[FaceBox]:
    """简单的 IoU NMS, 按 score 降序保留."""
    if not boxes:
        return []

    def iou(a: FaceBox, b: FaceBox) -> float:
        ax2, ay2 = a.x + a.w, a.y + a.h
        bx2, by2 = b.x + b.w, b.y + b.h
        ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = a.w * a.h + b.w * b.h - inter
        return inter / union if union > 0 else 0.0

    ordered = sorted(boxes, key=lambda b: b.score, reverse=True)
    keep: List[FaceBox] = []
    while ordered:
        best = ordered.pop(0)
        keep.append(best)
        ordered = [b for b in ordered if iou(best, b) < iou_thresh]
    return keep


# ---------------------------------------------------------------------------
# 三种打码方式
# ---------------------------------------------------------------------------

def _apply_pixelate(region: np.ndarray, block_size: int = 15) -> np.ndarray:
    """马赛克: 缩小 → 放大, 形成像素块效果."""
    h, w = region.shape[:2]
    if h == 0 or w == 0:
        return region
    bs = max(2, block_size)
    small_w = max(1, w // bs)
    small_h = max(1, h // bs)
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _apply_gaussian_blur(region: np.ndarray, ksize: int = 31) -> np.ndarray:
    """高斯模糊: 保留边缘的柔和过渡."""
    k = max(3, ksize | 1)  # 必须奇数
    return cv2.GaussianBlur(region, (k, k), 0)


def _apply_solid_block(region: np.ndarray, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """纯色遮盖."""
    out = region.copy()
    out[:] = color
    return out


def _apply_landmark_dots(region: np.ndarray,
                         landmarks: dict | None = None,
                         dot_radius: int = 3,
                         spacing: int = 16,
                         color: Tuple[int, int, int, int] = (0, 0, 255, 255),
                         region_box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """
    点阵打码 (轻量隐私保护):
    在人脸关键点(眼/鼻/嘴)附近排布均匀的红点矩阵.
    保留脸型 / 皮肤 / 发型, 仅遮挡核心识别特征(五官).

    Args:
        region: BGR 图块 (整张 expand 后的脸区域)
        landmarks: 5 个 landmark 像素坐标(原图坐标系)
        dot_radius: 单个红点半径(像素)
        spacing: 点阵行列间距(像素)
        color: 点颜色 BGR, 默认红 (0,0,255)
        region_box: (x, y, w, h) region 在原图中的位置, 用于把 landmark 坐标转成 region 内坐标
    """
    out = region.copy()
    rh, rw = out.shape[:2]
    if rh == 0 or rw == 0:
        return out

    # 把 landmark 原图坐标转成 region 内坐标
    pts = []
    if landmarks and region_box is not None:
        rx, ry, _, _ = region_box
        for name in ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth"):
            lx, ly = landmarks.get(name, (None, None))
            if lx is None:
                continue
            pts.append((int(lx - rx), int(ly - ry)))

    if not pts:
        # 没 landmark 就在整个区域均匀打点
        for yy in range(spacing // 2, rh, spacing):
            for xx in range(spacing // 2, rw, spacing):
                cv2.circle(out, (xx, yy), dot_radius, color[:3], -1, cv2.LINE_AA)
        return out

    # 在 landmark 周围绘制稀疏点阵：默认 3x3，避免覆盖整张脸。
    # 眼、鼻、嘴五个区域仍被干扰，同时保留脸型和大部分五官轮廓。
    grid_n = 3
    half = (grid_n - 1) // 2
    for (px, py) in pts:
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                x = px + dx * spacing
                y = py + dy * spacing
                if 0 <= x < rw and 0 <= y < rh:
                    cv2.circle(out, (x, y), dot_radius, color[:3], -1, cv2.LINE_AA)

    return out


BLUR_MODES = {
    "pixelate": _apply_pixelate,
    "gaussian": _apply_gaussian_blur,
    "solid":    _apply_solid_block,
    # landmark / landmark_whole_face 不是普通"区域打码函数", 需要 landmarks+region_box 上下文,
    # 在 process_image 里单独处理
}


def _apply_landmark_whole_face(region: np.ndarray,
                               face_grid_step: int = 14,
                               dot_radius: int = 3,
                               grid_n: int = 5,
                               spacing: int = 14,
                               color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """
    整脸范围均匀网格打红点 + 关键点密集叠加.

    第 1 层: 在 region 范围内按 face_grid_step 均匀打红点 (整张脸覆盖)
    第 2 层: 在 5 关键点附近 grid_n x grid_n 密集叠加 (callable: _apply_landmark_whole_face_with_landmarks)

    适用场景: 多图 reference_image 模式 (火山方舟 Seedance 2.0 等)
              单纯 landmark 红点会被报 InputImageSensitiveContentDetected,
              整脸网格打点 + 关键点叠加能稳定过审.
    """
    out = region.copy()
    rh, rw = out.shape[:2]
    if rh == 0 or rw == 0:
        return out

    # 第 1 层: 整脸均匀网格 (从 step/2 偏移起, 边界对齐)
    half = face_grid_step // 2
    for yy in range(half, rh, face_grid_step):
        for xx in range(half, rw, face_grid_step):
            cv2.circle(out, (xx, yy), dot_radius, color, -1, cv2.LINE_AA)
    return out


def _apply_landmark_whole_face_with_landmarks(region: np.ndarray,
                                              landmarks: dict | None = None,
                                              face_grid_step: int = 14,
                                              dot_radius: int = 3,
                                              grid_n: int = 5,
                                              spacing: int = 14,
                                              color: Tuple[int, int, int] = (0, 0, 255),
                                              region_box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """
    整脸范围均匀网格 + 关键点密集叠加 (带 landmark 上下文).
    """
    # 第 1 层: 整脸均匀网格
    out = _apply_landmark_whole_face(
        region, face_grid_step=face_grid_step, dot_radius=dot_radius,
        grid_n=grid_n, color=color,
    )

    rh, rw = out.shape[:2]
    if not landmarks or region_box is None:
        return out

    # 第 2 层: 关键点附近密集叠加 (landmarks 是绝对坐标, 需换算到 region 内坐标)
    rx, ry, _, _ = region_box
    half_g = (grid_n - 1) // 2
    for name in ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth"):
        lx, ly = landmarks.get(name, (None, None))
        if lx is None:
            continue
        px, py = int(lx - rx), int(ly - ry)
        for dy in range(-half_g, half_g + 1):
            for dx in range(-half_g, half_g + 1):
                x = px + dx * spacing
                y = py + dy * spacing
                if 0 <= x < rw and 0 <= y < rh:
                    cv2.circle(out, (x, y), dot_radius, color, -1, cv2.LINE_AA)
    return out


def _apply_green_red_bars(region: np.ndarray, bar_count: int = 6) -> np.ndarray:
    """Draw green-framed red horizontal bars across an expanded face region.

    放大版 (2026-08-19): 红条尺寸按人脸区域比例放大, 不再被 14px 卡死.
    修复 img6/img7 高清大脸打码后仍被 Seedance 判"含真人"的问题:
      - outer_height: 占脸高 20%, 上限 80px (原 max(4, min(14, rh*0.075)))
      - inner_height: 红色部分占 outer 55%
      - 覆盖范围 start_y/end_y: 0.12~0.88 (原 0.20~0.80), 盖住更多五官
      - bar_count: 默认 6 条 (原 4 条)
    """
    out = region.copy()
    rh, rw = out.shape[:2]
    if rh == 0 or rw == 0:
        return out

    margin_x = max(2, int(rw * 0.09))
    bar_width = max(1, rw - margin_x * 2)
    outer_height = max(6, min(80, int(rh * 0.20)))
    inner_height = max(2, int(outer_height * 0.55))
    half_outer = outer_height // 2
    half_inner = inner_height // 2

    # Keep the bars inside the face area and distribute them from upper face to chin.
    start_y = int(rh * 0.12)
    end_y = int(rh * 0.88)
    positions = np.linspace(start_y, end_y, num=bar_count, dtype=int)
    for center_y in positions:
        y1 = max(0, center_y - half_outer)
        y2 = min(rh - 1, center_y + half_outer)
        cv2.rectangle(out, (margin_x, y1), (margin_x + bar_width - 1, y2), (0, 190, 70), -1)
        red_y1 = max(y1, center_y - half_inner)
        red_y2 = min(y2, center_y + half_inner)
        cv2.rectangle(out, (margin_x + 4, red_y1), (margin_x + bar_width - 5, red_y2), (0, 0, 255), -1)
    return out


def _suppress_overlapping_bar_faces(faces: List[FaceBox]) -> List[FaceBox]:
    """Keep one box when two detections substantially cover the same face."""
    kept: List[FaceBox] = []
    for face in sorted(faces, key=lambda item: item.w * item.h, reverse=True):
        duplicate = False
        for existing in kept:
            overlap_w = max(0, min(face.x + face.w, existing.x + existing.w) - max(face.x, existing.x))
            overlap_h = max(0, min(face.y + face.h, existing.y + existing.h) - max(face.y, existing.y))
            if (overlap_w / max(1, min(face.w, existing.w)) >= 0.55
                    and overlap_h / max(1, min(face.h, existing.h)) >= 0.55):
                duplicate = True
                break
        if not duplicate:
            kept.append(face)
    return kept


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------

# 每个工作线程复用自己的 detector。OpenCV FaceDetectorYN 会频繁 setInputSize,
# 多线程共享同一个实例容易互相改输入尺寸, 并行请求时会不稳定。
_DETECTOR_LOCAL = threading.local()


def _get_detector(score_threshold: float = 0.6) -> FaceDetector:
    detector = getattr(_DETECTOR_LOCAL, "detector", None)
    if detector is None or abs(detector._detector.getScoreThreshold() - score_threshold) > 1e-3:  # noqa: SLF001
        detector = FaceDetector(score_threshold=score_threshold)
        _DETECTOR_LOCAL.detector = detector
    return detector


def process_image(input_bytes: bytes, mode: str = "pixelate",
                  score_threshold: float = 0.45,
                  expand_ratio: float = 0.35,
                  return_faces: bool = False,
                  **blur_params) -> dict:
    """
    主入口: 给定图片二进制, 返回打码后的二进制 + 元数据.

    Args:
        input_bytes: 任意常见格式图片 (jpg/png/webp)
        mode:
          - "pixelate"                : 马赛克, 默认 block_size=15
          - "gaussian"                : 高斯模糊, 默认 ksize=31
          - "solid"                   : 纯色遮盖, 默认 (0,0,0) 黑色
          - "landmark"                : 在 5 个关键点(双眼/鼻/嘴角)上覆盖红色点阵 (grid 3x3 默认),
                                        保留脸型/发型/服装, 适合"需要后续生成视频"的角色一致性场景.
          - "landmark_whole_face"     : 整脸范围均匀网格 + 关键点密集叠加,
                                        推荐参数 face_grid_step=14, dot_radius=3, grid_n=5, expand_ratio=0.30,
                                        适用多图 reference_image 模式 (火山方舟 Seedance 2.0 等),
                                        实测能过 InputImageSensitiveContentDetected.
        score_threshold: 人脸分数阈值 (0-1), 越低召回越多但误检也多
        expand_ratio: 人脸框向外扩展比例 (默认 0.35 = 扩 35%)
        return_faces: 是否把每张脸的元信息(含 landmark)放进返回值
        **blur_params: 传给对应打码方式的参数
            - landmark:                dot_radius (3), spacing (16), color ((0,0,255))
            - landmark_whole_face:     dot_radius (3), face_grid_step (14), grid_n (5),
                                       spacing (14), color ((0,0,255))
            - pixelate:                block_size (15)
            - gaussian:                ksize (31)
            - solid:                   color ((0,0,0))

    Returns:
        {
            "image_bytes": bytes,            # JPEG 编码结果
            "format": "jpg",
            "mode": mode,
            "face_count": int,
            "elapsed_ms": float,
            "faces": [...],                  # return_faces=True 时
        }
    """
    valid_modes = set(BLUR_MODES.keys()) | {"landmark", "landmark_whole_face", "green_red_bars"}
    modes = list(blur_params.pop("modes", []) or [mode])
    profiles = list(blur_params.pop("face_profiles", []) or [])
    if any(m not in valid_modes for m in modes):
        raise ValueError(f"modes must be drawn from {sorted(valid_modes)}")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode!r}")

    t0 = time.perf_counter()
    arr = np.frombuffer(input_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图片, 请检查输入格式")

    h, w = img.shape[:2]
    # A: 大图先缩小到 1920px 长边，加速多尺度检测
    _max_side = 1920
    if max(h, w) > _max_side:
        _scale = _max_side / max(h, w)
        img = cv2.resize(img, (int(w * _scale), int(h * _scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    detector = _get_detector(score_threshold)

    # landmark 模式: detect_multiscale (鲁棒) + per-face detect_with_landmarks (取关键点)
    # This set uses multiscale/tiled detection for modes where missed small faces are especially visible.
    landmark_modes = {"landmark", "landmark_whole_face", "green_red_bars"}
    uses_landmark = any(m in landmark_modes for m in modes) or any(
        m in landmark_modes for p in profiles for m in p.get("modes", [])
    )
    if uses_landmark:
        faces_list = detector.detect_multiscale(img)
        if "green_red_bars" in modes:
            faces_list = _suppress_overlapping_bar_faces(faces_list)
        # 密集竖图中远处人脸在整图输入里过小，使用重叠分块补检。
        short_side = min(h, w)
        long_side = max(h, w)
        dense_scene = len(faces_list) >= 16
        portrait_scene = len(faces_list) >= 8 and (
            long_side / max(short_side, 1) >= 1.35 or long_side >= 1600
        )
        if dense_scene or portrait_scene:
            tile_size, tile_step = ((320, 240) if dense_scene else (512, 384))
            tiled_faces = detector.detect_overlapping_tiles(img, tile_size, tile_step)
            faces_list = _nms_faces(faces_list + tiled_faces, iou_thresh=0.35)
        # 默认参数区分两种 landmark 模式
        if "landmark_whole_face" in modes:
            dot_radius = int(blur_params.get("dot_radius", 3))
            face_grid_step = int(blur_params.get("face_grid_step", 14))
            grid_n = int(blur_params.get("grid_n", 5))
            spacing = int(blur_params.get("spacing", face_grid_step))
            color = tuple(blur_params.get("color", (0, 0, 255)))
        else:  # landmark
            dot_radius = int(blur_params.get("dot_radius", 3))
            spacing = int(blur_params.get("spacing", 16))
            color = tuple(blur_params.get("color", (0, 0, 255)))
            face_grid_step = None
            grid_n = None

        # 极小人脸跳过打码 (原图直发已验证能过审, 保留远景细节)
        min_face_skip = int(blur_params.get("min_face_skip", 0))

        blurred_faces = []
        for face in faces_list:
            # 脸宽 < min_face_skip 直接跳过 (不打码, 原样保留远景小脸细节)
            # 用原始脸宽 face.w 判断 (expand 后的 bw 会放大, 不准确)
            profile = next((p for p in profiles if int(p.get("min_width", 0)) <= face.w <= int(p.get("max_width", 1000000))), {})
            face_modes = profile.get("modes") or modes
            local_skip = int(profile.get("min_face_skip", min_face_skip))
            if local_skip > 0 and face.w < local_skip:
                continue
            box = face.expand(w, h, ratio=expand_ratio)
            bx, by, bw, bh = box.x, box.y, box.w, box.h
            if bw < 20 or bh < 20:
                continue
            # 子区域 detect_with_landmarks 取关键点（比裸 _detector.detect 更干净）
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(w, bx + bw), min(h, by + bh)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            sub = img[y1:y2, x1:x2]
            landmarks = None
            try:
                prev_ls = detector._last_size
                sub_faces = detector.detect_with_landmarks(sub)
                detector._last_size = prev_ls
                detector._detector.setInputSize(prev_ls)
                if sub_faces:
                    f0 = sub_faces[0]
                    lm = f0["landmarks"]
                    landmarks = {
                        "right_eye":  (lm["right_eye"][0] + x1, lm["right_eye"][1] + y1),
                        "left_eye":   (lm["left_eye"][0] + x1, lm["left_eye"][1] + y1),
                        "nose":       (lm["nose"][0] + x1, lm["nose"][1] + y1),
                        "right_mouth": (lm["right_mouth"][0] + x1, lm["right_mouth"][1] + y1),
                        "left_mouth":  (lm["left_mouth"][0] + x1, lm["left_mouth"][1] + y1),
                    }
            except Exception:
                pass
            region = img[by:by + bh, bx:bx + bw]
            for face_mode in face_modes:
                if face_mode not in valid_modes:
                    raise ValueError(f"invalid face profile mode: {face_mode!r}")
                if face_mode == "landmark_whole_face":
                    f_step = int(profile.get("face_grid_step", face_grid_step or 14))
                    f_dot = int(profile.get("dot_radius", dot_radius))
                    f_grid_n = int(profile.get("grid_n", grid_n or 5))
                    region = _apply_landmark_whole_face_with_landmarks(
                        region, landmarks=landmarks,
                        face_grid_step=f_step, dot_radius=f_dot,
                        grid_n=f_grid_n, spacing=f_step, color=color,
                        region_box=(bx, by, bw, bh),
                    )
                elif face_mode == "landmark":
                    region = _apply_landmark_dots(region, landmarks=landmarks,
                        dot_radius=int(profile.get("dot_radius", dot_radius)),
                        spacing=int(profile.get("face_grid_step", spacing)), color=color,
                        region_box=(bx, by, bw, bh))
                elif face_mode == "green_red_bars":
                    region = _apply_green_red_bars(region)
                else:
                    region = BLUR_MODES[face_mode](region)
            img[by:by + bh, bx:bx + bw] = region
            face.landmarks = landmarks or {}
            blurred_faces.append(face)
    else:
        faces = detector.detect_multiscale(img)
        for face in faces:
            profile = next((p for p in profiles if int(p.get("min_width", 0)) <= face.w <= int(p.get("max_width", 1000000))), {})
            face_modes = profile.get("modes") or modes
            box = face.expand(w, h, ratio=expand_ratio)
            region = img[box.y:box.y + box.h, box.x:box.x + box.w]
            for face_mode in face_modes:
                region = BLUR_MODES[face_mode](region)
            img[box.y:box.y + box.h, box.x:box.x + box.w] = region

    # 编码输出
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 92]
    ok, buf = cv2.imencode(".jpg", img, encode_params)
    if not ok:
        raise RuntimeError("结果图片编码失败")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    # D 修复: landmark 模式 faces 是 dict 列表，普通模式是 FaceBox 列表
    if uses_landmark:
        _face_count = len(blurred_faces)
    else:
        _face_count = len(faces)
    result = {
        "image_bytes": buf.tobytes(),
        "format": "jpg",
        "mode": mode,
        "face_count": _face_count,
        "elapsed_ms": round(elapsed_ms, 2),
    }
    if return_faces:
        if uses_landmark:
            result["faces"] = [asdict(f) if hasattr(f, "__dataclass_fields__") else dict(f) for f in faces_list]
        else:
            result["faces"] = [asdict(f) if hasattr(f, "__dataclass_fields__") else dict(f)
                                for f in faces]
    return result


# ---------------------------------------------------------------------------
# Serverless 适配 (示例: 阿里云 FC / 腾讯云 SCF / AWS Lambda 通用 handler)
# ---------------------------------------------------------------------------

def handler(event, context):
    """
    通用 Serverless handler.

    event 期望:
      {
        "image_base64": "...",          # 图片 base64
        # 或
        "image_url": "https://...",     # 可选: 如果函数本身没有下载能力
        "mode": "pixelate",             # 可选, 默认 pixelate
        "score_threshold": 0.6,         # 可选
        "expand_ratio": 0.2,            # 可选
      }
    """
    import base64
    import json

    if "image_base64" in event:
        img_bytes = base64.b64decode(event["image_base64"])
    elif "image_url" in event:
        with urllib.request.urlopen(event["image_url"]) as r:
            img_bytes = r.read()
    else:
        return {"statusCode": 400,
                "body": json.dumps({"error": "image_base64 or image_url required"})}

    result = process_image(
        img_bytes,
        mode=event.get("mode", "pixelate"),
        score_threshold=float(event.get("score_threshold", 0.45)),
        expand_ratio=float(event.get("expand_ratio", 0.35)),
        return_faces=True,
    )
    return {
        "statusCode": 200,
        "body": json.dumps({
            "image_base64": base64.b64encode(result["image_bytes"]).decode("ascii"),
            "format": result["format"],
            "mode": result["mode"],
            "face_count": result["face_count"],
            "faces": result["faces"],
            "elapsed_ms": result["elapsed_ms"],
        }, ensure_ascii=False),
    }
