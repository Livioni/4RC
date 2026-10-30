#!/usr/bin/env python3
"""Calculate resized/padded camera intrinsics from the first Markdown table.

Run from the project root: python scripts/resize_camera_intrinsics.py
Only the Python standard library is required. The original table is retained;
the generated section in the Markdown and the output JSON are reproducible.
"""

import argparse
import ast
from decimal import Decimal
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- resized-intrinsics:begin -->"
END = "<!-- resized-intrinsics:end -->"
HALF = Decimal("0.5")


def read_robotwin_constants():
    tree = ast.parse((ROOT / "arc/datasets/robotwin.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "RoboTwin4RC")
    return {target.id: node.value.value
            for node in cls.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            for target in node.targets if isinstance(target, ast.Name)}


def read_intrinsics(markdown):
    result = {}
    for line in markdown.splitlines():
        if not line.lstrip().startswith("|"):
            if result:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or not cells[0].lower().startswith("cam_"):
            continue
        camera = cells[0].lower()  # cam_Left_wrist in the source is cam_left_wrist in HDF5.
        if camera in result:
            raise ValueError(f"Duplicate camera in the original table: {camera}")
        result[camera] = (cells[0], tuple(Decimal(value) for value in cells[1:]))
    if not result:
        raise ValueError("No camera/fx/fy/cx/cy table found")
    for camera, (_, values) in result.items():
        if not all(value.is_finite() for value in values) or min(values[:2]) <= 0:
            raise ValueError(f"Invalid intrinsic values: {camera}")
    return result


def pack(values):
    fx, fy, cx, cy = map(float, values)
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]}


def table(cameras, field):
    lines = ["| 相机 | fx | fy | cx | cy |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for camera, data in cameras.items():
        values = data[field]
        lines.append("| " + camera + " | " + " | ".join(
            f"{values[key]:.4f}" for key in ("fx", "fy", "cx", "cy")) + " |")
    return "\n".join(lines)


def main():
    config = read_robotwin_constants()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "scripts/相机内参.md")
    parser.add_argument("--source-width", type=int, default=640)
    parser.add_argument("--source-height", type=int, default=480)
    parser.add_argument("--width", type=int, default=config["SOURCE_WIDTH"])
    parser.add_argument("--height", type=int, default=config["SOURCE_HEIGHT"])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "datasets/4_cams_lerobot_v21/meta/intrinsics_from_markdown.json")
    parser.add_argument("--no-update-markdown", action="store_true")
    args = parser.parse_args()
    if min(args.source_width, args.source_height, args.width, args.height) <= 0:
        parser.error("All image dimensions must be positive")
    original_text = args.input.read_text(encoding="utf-8")
    original = read_intrinsics(original_text)
    sx = Decimal(args.width) / args.source_width
    sy = Decimal(args.height) / args.source_height
    left, right = config["PAD_LEFT"], config["PAD_RIGHT"]
    top, bottom = config["PAD_TOP"], config["PAD_BOTTOM"]
    cameras = {}
    for camera, (source_name, values) in original.items():
        fx, fy, cx, cy = values
        resized = (sx * fx, sy * fy, (cx + HALF) * sx - HALF, (cy + HALF) * sy - HALF)
        padded = (resized[0], resized[1], resized[2] + left, resized[3] + top)
        cameras[camera] = {"source_camera_name": source_name, "source": pack(values),
                           "resized": pack(resized), "model_padded": pack(padded),
                           "direct_scale_reference": pack((fx * sx, fy * sy, cx * sx, cy * sy))}
    result = {
        "source": str(args.input.resolve()),
        "source_size_wh": [args.source_width, args.source_height],
        "resized_size_wh": [args.width, args.height],
        "scale_xy": [float(sx), float(sy)],
        "convention": "OpenCV pixel centers: c'=(c+0.5)*scale-0.5; f'=f*scale",
        "source_pixel_to_resized_pixel": [[float(sx), 0.0, float((sx - 1) / 2)],
                                           [0.0, float(sy), float((sy - 1) / 2)],
                                           [0.0, 0.0, 1.0]],
        "model_padded_size_wh": [args.width + left + right, args.height + top + bottom],
        "padding_lrtb": [left, right, top, bottom],
        "padded_intrinsics_formula": "fx,fy unchanged; cx+=pad_left; cy+=pad_top",
        "source_precision": "Values are taken from the Markdown table, rounded to three decimals; "
                            "calibration.json uses the HDF5 full-precision calibration.",
        "depth": "These intrinsics also apply to depth already aligned to rectified RGB.",
        "cameras": cameras,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    if not args.no_update_markdown:
        width_pad, height_pad = result["model_padded_size_wh"]
        section = (
            f"{BEGIN}\n\n"
            f"## 分辨率变更后的内参（自动生成）\n\n"
            f"原始分辨率为 **{args.source_width}×{args.source_height}**，"
            f"LeRobot 数据分辨率为 **{args.width}×{args.height}**。"
            "下表以本文原始表格为输入，与转换脚本使用相同的 OpenCV 像素中心约定：\n\n"
            "```text\n"
            "sx = target_width / source_width; sy = target_height / source_height\n"
            "fx' = fx * sx; fy' = fy * sy\n"
            "cx' = (cx + 0.5) * sx - 0.5\n"
            "cy' = (cy + 0.5) * sy - 0.5\n"
            "K'  = [[fx', 0, cx'], [0, fy', cy'], [0, 0, 1]]\n"
            "```\n\n"
            + table(cameras, "resized") + "\n\n"
            f"### 4RC 模型补边后：{width_pad}×{height_pad}\n\n"
            f"项目 `arc/datasets/robotwin.py` 使用左/右各 {left}/{right} 像素、"
            f"上/下各 {top}/{bottom} 像素的补边。"
            f"焦距保持不变，主点加上左、上补边量：`cx'' = cx' + {left}`，"
            f"`cy'' = cy' + {top}`。\n\n"
            + table(cameras, "model_padded") + "\n\n"
            "### 直接缩放约定的参考值\n\n"
            "若下游工具明确采用 `cx' = cx * sx, cy' = cy * sy`，可用下表。"
            "本次转换脚本采用上面的像素中心约定；缩小一半时，"
            "该约定的 `cx, cy` 比直接乘 0.5 各小 0.25 像素。\n\n"
            + table(cameras, "direct_scale_reference") + "\n\n"
            "深度已对齐到校正后的 RGB，可使用相同的输出内参。"
            "`cam_Left_wrist` 已按数据集名称统一为 `cam_left_wrist`。"
            "本文原始数值只保留了三位小数，计算结果与数据集 `meta/calibration.json` "
            "中基于 HDF5 完整精度的结果存在微小舍入差异。\n\n"
            "重新生成：`python scripts/resize_camera_intrinsics.py`。"
            "完整矩阵保存在 `datasets/4_cams_lerobot_v21/meta/intrinsics_from_markdown.json`。\n\n"
            f"{END}"
        )
        if BEGIN in original_text:
            before, tail = original_text.split(BEGIN, 1)
            if END not in tail:
                raise ValueError("Generated Markdown section is missing its end marker")
            _, after = tail.split(END, 1)
            updated = before + section + after
        else:
            updated = original_text.rstrip() + "\n\n" + section + "\n"
        args.input.write_text(updated, encoding="utf-8")
    print(table(cameras, "resized"))
    print(f"\nJSON: {args.output}")


if __name__ == "__main__":
    main()
