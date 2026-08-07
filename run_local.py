"""
本地批处理入口: 对指定目录下的图片逐张生成三种打码结果.

用法:
    python run_local.py                          # 默认处理 ../渠道5-智能过人脸/示例图片.jpg
    python run_local.py path/to/img1.jpg [more...]
    python run_local.py --dir path/to/img_dir/

输出:
    outputs/<原名>_pixelate.jpg
    outputs/<原名>_gaussian.jpg
    outputs/<原名>_solid.jpg
    outputs/<原名>_meta.json   检测元数据 (face_count, bboxes, 耗时)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from face_blur import process_image

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "示例图片.jpg"
OUTPUT_DIR = HERE / "outputs"
MODES = ["pixelate", "gaussian", "solid", "landmark"]


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    inputs: list[Path] = []
    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            sys.exit(f"--dir 指定的路径不存在或不是目录: {d}")
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            inputs.extend(sorted(d.glob(ext)))
    inputs.extend(Path(p) for p in args.inputs)
    if not inputs and DEFAULT_INPUT.exists():
        inputs = [DEFAULT_INPUT]
    if not inputs:
        sys.exit("未提供任何输入图片, 且默认示例图片也不存在")
    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="批量本地人脸打码")
    parser.add_argument("inputs", nargs="*", help="图片路径, 可多个")
    parser.add_argument("--dir", help="图片目录, 会读取目录下所有 jpg/png/webp/bmp")
    parser.add_argument("--score", type=float, default=0.45,
                        help="人脸分数阈值 0-1, 默认 0.45 (越低召回越多)")
    parser.add_argument("--expand", type=float, default=0.35,
                        help="人脸框向外扩比例, 默认 0.35 (避免边缘漏脸)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = collect_inputs(args)
    print(f"[run_local] 待处理图片: {len(inputs)} 张 -> 输出目录 {OUTPUT_DIR}")

    total_faces = 0
    for path in inputs:
        if not path.is_file():
            print(f"  [skip] 不是文件: {path}")
            continue
        data = path.read_bytes()
        print(f"\n--- {path.name} ({len(data)/1024:.1f} KB) ---")
        meta: dict = {"source": str(path), "results": {}}
        for mode in MODES:
            try:
                r = process_image(data, mode=mode,
                                  score_threshold=args.score,
                                  expand_ratio=args.expand,
                                  return_faces=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [{mode}] FAIL: {e}")
                meta["results"][mode] = {"error": str(e)}
                continue
            out_path = OUTPUT_DIR / f"{path.stem}_{mode}.jpg"
            out_path.write_bytes(r["image_bytes"])
            meta["results"][mode] = {
                "face_count": r["face_count"],
                "faces": r["faces"],
                "elapsed_ms": r["elapsed_ms"],
                "output": str(out_path),
            }
            total_faces += r["face_count"]
            print(f"  [{mode:9s}] face_count={r['face_count']:<2d} "
                  f"elapsed={r['elapsed_ms']:>6.1f}ms -> {out_path.name}")

        meta_path = OUTPUT_DIR / f"{path.stem}_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"\n[run_local] done. 共处理 {len(inputs)} 张, "
          f"累计检测人脸 {total_faces} 张, 输出在 {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
