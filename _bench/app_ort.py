"""
ORT 版 API 服务启动器 (测试用): 通过 monkeypatch 把 face_blur.FaceDetector
替换为 onnxruntime 实现, 其余逻辑(缓存/下载/API)完全复用 app.py.

用法: uvicorn app_ort:app --port 8001 --workers 1
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "_bench"))

import face_blur as fb
from _bench.ort_detector import OrtYuNetDetector


class ORTDetectorWrapper:
    """对齐 face_blur.FaceDetector 接口的 ORT 版 (含 landmark 多尺度)."""

    def __init__(self, score_threshold: float = 0.6,
                 nms_threshold: float = 0.3, top_k: int = 50):
        self._detector = OrtYuNetDetector(
            str(fb.MODEL_PATH),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )
        self._threshold = score_threshold

    # -- face_blur._get_detector 需要 getScoreThreshold 兼容 --
    def getScoreThreshold(self):
        return self._threshold

    # -- face_blur.FaceDetector 兼容接口 --
    def detect(self, img_bgr):
        return [fb.FaceBox(x, y, w, h, s, "ort")
                for (x, y, w, h, s) in self._detector.detect(img_bgr)]

    def detect_with_landmarks(self, img_bgr):
        out = []
        for d in self._detector.detect_with_landmarks(img_bgr):
            out.append({
                "x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"],
                "score": d["score"],
                "landmarks": {k: tuple(v) for k, v in d["landmarks"].items()},
            })
        return out

    def detect_multiscale(self, img_bgr,
                          target_long_sides=(640, 1600),
                          use_haar_fallback=False):
        boxes = self._detector.detect_multiscale(img_bgr, target_long_sides)
        return [fb.FaceBox(x, y, w, h, s, "ort")
                for (x, y, w, h, s) in boxes]

    def detect_haar_fallback(self, img_bgr):
        return []


# ---- monkeypatch: 让 face_blur 的 _get_detector 返回 ORT 版 ----
fb.FaceDetector = ORTDetectorWrapper
fb._DETECTOR_LOCAL = __import__("threading").local()

# ---- 复用 app.py 的完整应用 (此时它 import 的 process_image 会用 patch 后的 FaceDetector) ----
import app as app_module  # noqa: E402

app = app_module.app

# 启动入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, workers=1)
