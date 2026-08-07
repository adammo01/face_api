"""
本地静态图片服务器: 供 /api/face_blur 的 image_url 参数使用
用法: python _bench/img_server.py  (监听 18999 端口)
"""
from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        pass  # 静默


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18999)
    ap.add_argument("--bind", default="127.0.0.1")
    args = ap.parse_args()
    with socketserver.ThreadingTCPServer((args.bind, args.port), Handler) as httpd:
        print(f"[img_server] serving {ROOT} on http://{args.bind}:{args.port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
