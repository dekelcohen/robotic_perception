import argparse
import ast
import json
import os
from typing import List, Any
from PIL import Image

# Ensure provider can be imported
from features_markers.bbox_providers.moondream_bbox_provider import MoondreamVLMProvider

def parse_spatial_refs(s: str) -> List[List[float]]:
    s = s.strip()
    if not s:
        return []
    # Wrap into a list if not already
    text = s
    if not (text.startswith('[') and text.endswith(']')) or text.count('[') == 1:
        text = f'[{s}]'
    try:
        data = ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Failed to parse --spatial_refs='{s}'. Expected format like: "
                         f"\"[x1,y1],[x2,y2],[x3,y3,x4,y4]\". Error: {e}")
    if not isinstance(data, list):
        raise ValueError("--spatial_refs must parse to a list of lists.")
    refs: List[List[float]] = []
    for item in data:
        if not isinstance(item, (list, tuple)):
            raise ValueError(f"Each spatial ref must be a list/tuple. Got: {item}")
        if len(item) not in (2, 4):
            raise ValueError(f"Each spatial ref must have length 2 (point) or 4 (box). Got: {item}")
        try:
            refs.append([float(v) for v in item])
        except Exception:
            raise ValueError(f"Spatial ref values must be numeric. Got: {item}")
    return refs

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Moondream segment CLI test")
    parser.add_argument('--image', required=True, help='Path to input PNG image')
    parser.add_argument('--spatial_refs', default='', help='Refs string: "[x1,y1],[x2,y2],[x3,y3,x4,y4]"')
    parser.add_argument('--prompt', default='', help='Text prompt (optional)')
    args = parser.parse_args(argv)

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return 2
    if not args.spatial_refs and not args.prompt:
        print("Error: one of --spatial_refs or --prompt must be provided.")
        return 2

    image = Image.open(args.image).convert('RGB')

    spatial_refs: List[List[float]] = []
    if args.spatial_refs:
        spatial_refs = parse_spatial_refs(args.spatial_refs)

    prompt = args.prompt or 'object'
    provider = MoondreamVLMProvider()
    try:
        result = provider.segment(image, [prompt], spatial_refs=spatial_refs if spatial_refs else None)
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return 1

    print(json.dumps(result, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

