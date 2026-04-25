"""
Runnable sample for MoondreamVLMProvider.segment.

Usage examples:
  python -m features_markers.bbox_providers.tests.sample_moondream_segment_main --image "E:\Robotics\VLM_Robotics\language-models-trajectory-generators\outputs\bugs\door_rgb_image_head.png" --classes "door handle"

  python features_markers/bbox_providers/tests/sample_moondream_segment_main.py \
      --image path/to/image.jpg -c cup -c bottle

Notes:
- Requires environment variable `MOONDREAM_API_KEY` to be set.
"""
import argparse
import json
import numpy as np
import os
import sys
from typing import List

from features_markers.bbox_providers.moondream_bbox_provider import MoondreamVLMProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoondreamVLMProvider.segment on an image.")
    parser.add_argument("--image", required=True, help="Path to the image file (jpg/png).")
    parser.add_argument(
        "--classes",
        "-c",
        action="append",
        help=(
            "Class prompts (repeat -c for multiple) or provide a single "
            "comma-separated string (e.g., 'cup,bottle')."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def normalize_classes(arg_classes: List[str] | None) -> List[str]:
    if not arg_classes:
        return []
    prompts: List[str] = []
    for entry in arg_classes:
        if entry is None:
            continue
        if "," in entry:
            prompts.extend([s.strip() for s in entry.split(",") if s.strip()])
        else:
            prompts.append(entry.strip())
    return prompts


def main() -> int:
    # Phase: validate environment
    api_key = os.environ.get("MOONDREAM_API_KEY")
    if not api_key:
        print("ERROR: MOONDREAM_API_KEY is not set in the environment.", file=sys.stderr)
        return 2

    args = parse_args()
    prompts = normalize_classes(args.classes)
    if not prompts:
        print("ERROR: No class prompts provided. Use -c or --classes.", file=sys.stderr)
        return 2

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
        return 2

    print("[Moondream Sample] Starting segmentation...")
    print(f"  image: {image_path}")
    print(f"  prompts: {prompts}")

    try:
        provider = MoondreamVLMProvider()
        result = provider.segment(image_path, prompts)
    except Exception as e:
        # Robustness: surface errors clearly to callers.
        print(f"ERROR: Segmentation failed: {e}", file=sys.stderr)
        return 1

    def _json_default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    dump = json.dumps(result, indent=2 if args.pretty else None, default=_json_default)
    print("[Moondream Sample] Result:")
    print(dump)
    print("[Moondream Sample] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



