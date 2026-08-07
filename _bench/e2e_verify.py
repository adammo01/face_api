"""端到端功能验证: 5 种模式各调一次 API"""
import json
import sys
import requests

BASE = "http://127.0.0.1:8000"
IMG = "http://127.0.0.1:18999/示例图片.jpg"

for mode in ("gaussian", "pixelate", "solid", "landmark", "landmark_whole_face"):
    r = requests.post(f"{BASE}/api/face_blur",
                      json={"image_url": IMG, "mode": mode,
                            "score_threshold": 0.6, "expand_ratio": 0.30},
                      timeout=120)
    d = r.json()
    print(f"{mode:20s} http={r.status_code} blocked={d.get('blocked')} "
          f"faces={d.get('face_count')} elapsed={d.get('elapsed_ms')}ms "
          f"output_url={d.get('output_url', '')[:80]}")

# 验证输出图可访问且是有效 JPEG
r = requests.post(f"{BASE}/api/face_blur",
                  json={"image_url": IMG, "mode": "gaussian",
                        "score_threshold": 0.6, "expand_ratio": 0.30},
                  timeout=120)
d = r.json()
if d.get("output_url"):
    out = requests.get(d["output_url"], timeout=30)
    print(f"\noutput_url HTTP {out.status_code}, {len(out.content)} bytes, "
          f"magic={out.content[:2]}")
    assert out.content[:2] == b"\xff\xd8", "不是有效 JPEG!"
    print("输出图验证 OK: 有效 JPEG")
