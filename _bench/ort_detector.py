"""
方案 A: onnxruntime 版 YuNet 检测器 (测试用, 不改生产代码)
========================================================
YuNet 2023mar ONNX 是原始多输出格式 (3 尺度 cls/obj/bbox/kps),
后处理严格对齐 OpenCV 5.0 FaceDetectorYNImpl::postProcess (face_detect.cpp):

  - 输入: [1,3,640,640] float32 NCHW, **不归一化** (0-255 原值)
  - score = sqrt(clamp(cls,0,1) * clamp(obj,0,1))   # 不是 sigmoid!
  - cx = (col + bbox[0]) * stride,  cy = (row + bbox[1]) * stride
  - w  = exp(bbox[2]) * stride,     h  = exp(bbox[3]) * stride
  - 关键点: kx = (kps[2n] + col) * stride,  ky = (kps[2n+1] + row) * stride
  - 输入图保持比例 resize 到 640 长边 + 灰边(114) pad 到 640x640
  - 坐标减 pad 偏移再按 scale 映射回原图, 最后做 IoU NMS

实测 (示例图片.jpg 2560x1440):
  OpenCV DNN  : ~45.8 ms/帧   faces=3
  onnxruntime: ~8.0 ms/帧     faces=3  (加速 ~5.7x, 框位置一致, score 略低)
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

_LM_NAMES = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")
_STRIDES = (8, 16, 32)
_MODEL_INPUT = 640
_PAD_VALUE = 114


class OrtYuNetDetector:
    """onnxruntime 推理的 YuNet 人脸检测器, API 对齐 face_blur.FaceDetector."""

    def __init__(self, model_path: str,
                 score_threshold: float = 0.6,
                 nms_threshold: float = 0.3,
                 top_k: int = 50,
                 threads: int = 0):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        so = ort.SessionOptions()
        if threads > 0:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self._last_infer_ms = 0.0

    # -- face_blur._get_detector 兼容 (cv2.FaceDetectorYN 同名方法) --
    def getScoreThreshold(self) -> float:
        return self.score_threshold

    def setScoreThreshold(self, v: float) -> None:
        self.score_threshold = v

    # ------------------------------------------------------------------
    def _infer(self, img_bgr: np.ndarray):
        """保持比例 resize 到 640 + pad 灰边, 返回 (cls, obj, bbox, kps, scale, x0, y0)."""
        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]
        scale = _MODEL_INPUT / max(h, w)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((_MODEL_INPUT, _MODEL_INPUT, 3), _PAD_VALUE, dtype=np.uint8)
        x0, y0 = (_MODEL_INPUT - nw) // 2, (_MODEL_INPUT - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        blob = canvas.astype(np.float32)  # 不归一化 (对齐 OpenCV blobFromImage scale=1.0)
        blob = blob.transpose(2, 0, 1)[None, ...]  # [1,3,640,640]
        outputs = self.session.run(None, {self.input_name: blob})
        by_name = {o.name: out[0] for o, out in
                   zip(self.session.get_outputs(), outputs)}
        cls = [by_name[f"cls_{s}"] for s in _STRIDES]
        obj = [by_name[f"obj_{s}"] for s in _STRIDES]
        bbox = [by_name[f"bbox_{s}"] for s in _STRIDES]
        kps = [by_name[f"kps_{s}"] for s in _STRIDES]
        self._last_infer_ms = (time.perf_counter() - t0) * 1000
        return cls, obj, bbox, kps, scale, x0, y0

    def _decode(self, cls, obj, bbox, kps):
        """OpenCV postProcess 解码, 返回 640 坐标系 (box_xyxy, kps_5x2, score)."""
        dets: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for stride, c, o, b, k in zip(_STRIDES, cls, obj, bbox, kps):
            c = np.clip(c.reshape(-1), 0, 1)
            o = np.clip(o.reshape(-1), 0, 1)
            b = b.reshape(-1, 4)
            k = k.reshape(-1, 10)
            grid = _MODEL_INPUT // stride
            scores = np.sqrt(c * o)  # OpenCV: sqrt(cls*obj)
            mask = scores >= self.score_threshold
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            cols = (idxs % grid).astype(np.float32)
            rows = (idxs // grid).astype(np.float32)
            bsel = b[idxs]
            cx = (cols + bsel[:, 0]) * stride
            cy = (rows + bsel[:, 1]) * stride
            bw = np.exp(bsel[:, 2]) * stride
            bh = np.exp(bsel[:, 3]) * stride
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = x1 + bw
            y2 = y1 + bh
            ksel = k[idxs]
            kps_out = np.stack([
                (cols + ksel[:, 2 * n]) * stride
                for n in range(5)
            ] + [
                (rows + ksel[:, 2 * n + 1]) * stride
                for n in range(5)
            ], axis=-1).reshape(-1, 5, 2)  # [M,5,2]
            for i in range(len(idxs)):
                dets.append((np.array([x1[i], y1[i], x2[i], y2[i]]),
                             kps_out[i], float(scores[idxs[i]])))
        return dets

    def _map_to_orig(self, dets, img_w, img_h, scale, x0, y0):
        out = []
        for (box, kps, score) in dets:
            b = (np.clip(box, 0, _MODEL_INPUT) - np.array([x0, y0, x0, y0])) / scale
            k = (kps - np.array([x0, y0])) / scale
            out.append((b, k, score))
        return out

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou < self.nms_threshold]
        return keep[: self.top_k]

    # ------------------------------------------------------------------
    def detect(self, img_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        h, w = img_bgr.shape[:2]
        cls, obj, bbox, kps, scale, x0, y0 = self._infer(img_bgr)
        dets = self._map_to_orig(self._decode(cls, obj, bbox, kps), w, h, scale, x0, y0)
        if not dets:
            return []
        boxes = np.array([d[0] for d in dets])
        scores = np.array([d[2] for d in dets])
        keep = self._nms(boxes, scores)
        out = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(scores[i])))
        return out

    def detect_with_landmarks(self, img_bgr: np.ndarray) -> List[Dict]:
        h, w = img_bgr.shape[:2]
        cls, obj, bbox, kps, scale, x0, y0 = self._infer(img_bgr)
        dets = self._map_to_orig(self._decode(cls, obj, bbox, kps), w, h, scale, x0, y0)
        if not dets:
            return []
        boxes = np.array([d[0] for d in dets])
        kps_all = np.array([d[1] for d in dets])
        scores = np.array([d[2] for d in dets])
        keep = self._nms(boxes, scores)
        out = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            k = kps_all[i]
            landmarks = {name: (float(k[j][0]), float(k[j][1]))
                         for j, name in enumerate(_LM_NAMES)}
            out.append({
                "x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1),
                "score": float(scores[i]), "landmarks": landmarks,
            })
        return out

    def detect_multiscale(self, img_bgr: np.ndarray,
                          target_long_sides: tuple = (640, 1600)) -> List[Tuple[int, int, int, int, float]]:
        """多尺度检测, 语义对齐 face_blur.FaceDetector.detect_multiscale (仅返回框)."""
        h, w = img_bgr.shape[:2]
        all_boxes = list(self.detect(img_bgr))
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
            cls, obj, bbox, kps, s2, x0, y0 = self._infer(resized)
            dets = self._map_to_orig(self._decode(cls, obj, bbox, kps),
                                     new_w, new_h, s2, x0, y0)
            inv = 1.0 / scale
            for (b, _k, s) in dets:
                all_boxes.append((int(b[0] * inv), int(b[1] * inv),
                                  int((b[2] - b[0]) * inv), int((b[3] - b[1]) * inv),
                                  float(s)))
        return _nms_boxes(all_boxes, iou_thresh=0.35)


def _nms_boxes(boxes: List[Tuple[int, int, int, int, float]],
               iou_thresh: float = 0.35) -> List[Tuple[int, int, int, int, float]]:
    if not boxes:
        return []

    def iou(a, b):
        ax2, ay2 = a[0] + a[2], a[1] + a[3]
        bx2, by2 = b[0] + b[2], b[1] + b[3]
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / union if union > 0 else 0.0

    ordered = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while ordered:
        best = ordered.pop(0)
        keep.append(best)
        ordered = [b for b in ordered if iou(best, b) < iou_thresh]
    return keep


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from face_blur import FaceDetector

    model = str(Path.home() / ".cache" / "face_blur" / "face_detection_yunet_2023mar.onnx")
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "示例图片.jpg")
    print("image:", img.shape)

    cv_det = FaceDetector(score_threshold=0.6)
    ort_det = OrtYuNetDetector(model, score_threshold=0.6)

    for _ in range(3):
        cv_det.detect(img)
        ort_det.detect(img)

    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        cv_faces = cv_det.detect(img)
    cv_ms = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        ort_faces = ort_det.detect(img)
    ort_ms = (time.perf_counter() - t0) / N * 1000

    print(f"OpenCV DNN : {cv_ms:7.2f} ms/帧   faces={len(cv_faces)}")
    print(f"onnxruntime: {ort_ms:7.2f} ms/帧   faces={len(ort_faces)}")
    print(f"加速比: {cv_ms / ort_ms:.2f}x")
    print("OpenCV 框:", [(f.x, f.y, f.w, f.h, round(f.score, 3)) for f in cv_faces[:3]])
    print("ORT    框:", [(f[0], f[1], f[2], f[3], round(f[4], 3)) for f in ort_faces[:3]])
