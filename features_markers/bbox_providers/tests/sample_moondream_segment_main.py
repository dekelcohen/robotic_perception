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
from pathlib import Path
import sys
from typing import List

from PIL import Image

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
    parser.add_argument(
        "--overlay-output",
        help=(
            "Optional output path for a PNG with transparent segmentation overlay. "
            "Default: next to --image with suffix '_moondream_seg_overlay.png'."
        ),
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


def _extract_seg_masks(result: object) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    if not isinstance(result, dict):
        return masks

    direct_mask = result.get("seg_mask")
    if direct_mask is not None:
        masks.append(np.asarray(direct_mask))

    predictions = result.get("predictions")
    if isinstance(predictions, list):
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            pred_mask = pred.get("seg_mask")
            if pred_mask is None:
                continue
            masks.append(np.asarray(pred_mask))
    return masks


def _normalize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"seg_mask must be 2D. Got shape={arr.shape}")
    if arr.shape == (height, width):
        return (arr > 0).astype(np.uint8)
    if arr.shape == (width, height):
        return (arr.T > 0).astype(np.uint8)
    raise ValueError(f"seg_mask shape {arr.shape} does not match image size {(height, width)}.")


def _save_mask_overlay_png(image_path: str, masks: List[np.ndarray], output_path: str | None) -> str:
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    combined = np.zeros((height, width), dtype=np.uint8)

    for mask in masks:
        norm_mask = _normalize_mask(mask, width, height)
        combined = np.maximum(combined, norm_mask)

    alpha = (combined * 120).astype(np.uint8)
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[..., 1] = 255
    overlay[..., 3] = alpha

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    composite = Image.alpha_composite(image, overlay_img)

    if output_path:
        out_path = Path(output_path)
    else:
        src_path = Path(image_path)
        out_path = src_path.with_name(f"{src_path.stem}_moondream_seg_overlay.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path, format="PNG")
    return str(out_path)


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
    #dump = json.dumps(result, indent=2 if args.pretty else None, default=_json_default)    
    #print(dump)
        
    masks = _extract_seg_masks(result)
            
    if not masks:
        print("ERROR: No seg_mask found in result.", file=sys.stderr)
        return 1

    if len(masks) > 1:
        print(
            f"[Moondream Sample] Found {len(masks)} seg_mask candidates; using the first one for overlay."
        )

    for idx, pred in enumerate(result['predictions']):
        print(f'Moondream segment {idx} svg path:', pred['path'])
        
    try:
        overlay_path = _save_mask_overlay_png(image_path, [masks[0]], args.overlay_output)
    except Exception as e:
        print(f"ERROR: Failed to create overlay PNG: {e}", file=sys.stderr)
        return 1

    print(f"[Moondream Sample] Saved mask overlay PNG: {overlay_path}")
    print("[Moondream Sample] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



