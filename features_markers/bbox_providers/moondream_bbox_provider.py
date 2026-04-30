from __future__ import annotations

from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw
import os
import numpy as np

# Specialized SVG path parser (preferred)
from svgpathtools import parse_path

from .bbox_providers import BBoxProvider


def _ensure_pil(image_or_path: Any) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        return image_or_path
    if isinstance(image_or_path, (bytes, bytearray)):
        from io import BytesIO
        return Image.open(BytesIO(image_or_path)).convert("RGB")
    if isinstance(image_or_path, str):
        return Image.open(image_or_path).convert("RGB")
    try:
        return Image.open(image_or_path).convert("RGB")  # type: ignore[arg-type]
    except Exception as e:
        raise ValueError(f"Unsupported image input type: {type(image_or_path).__name__}") from e



def _normalize_ref(ref: List[float], W: int, H: int):
    """
    Normalize a spatial reference to (kind, values) with kind in {"point", "bbox"}.
    - Accepts normalized refs (0..1) or pixel refs within image bounds.
    - Raises ValueError with a descriptive message if the ref length is not 2 or 4,
      or if any coordinate is out of bounds.
    """
    try:
        vals = [float(v) for v in ref]
    except Exception:
        raise ValueError(f"spatial_ref must contain numeric values; got: {ref}")

    if len(vals) == 2:
        x, y = vals
        is_pixel = max(abs(x), abs(y)) > 1.0
        if is_pixel:
            if not (0.0 <= x <= float(W)):
                raise ValueError(f"Point x out of range: {x} not in [0,{W}] for image width={W}.")
            if not (0.0 <= y <= float(H)):
                raise ValueError(f"Point y out of range: {y} not in [0,{H}] for image height={H}.")
            x = x / float(W)
            y = y / float(H)
        else:
            if not (0.0 <= x <= 1.0):
                raise ValueError(f"Normalized point x out of range: {x} not in [0,1].")
            if not (0.0 <= y <= 1.0):
                raise ValueError(f"Normalized point y out of range: {y} not in [0,1].")
        return ("point", [x, y])

    if len(vals) == 4:
        x1, y1, x2, y2 = vals
        is_pixel = max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.0
        if is_pixel:
            if not (0.0 <= x1 <= float(W)) or not (0.0 <= x2 <= float(W)):
                raise ValueError(f"Box x values out of range: [{x1}, {x2}] not within [0,{W}] for image width={W}.")
            if not (0.0 <= y1 <= float(H)) or not (0.0 <= y2 <= float(H)):
                raise ValueError(f"Box y values out of range: [{y1}, {y2}] not within [0,{H}] for image height={H}.")
            x1 /= float(W); x2 /= float(W)
            y1 /= float(H); y2 /= float(H)
        else:
            for label, v in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"Normalized {label} out of range: {v} not in [0,1].")
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
        return ("bbox", [x_min, y_min, x_max, y_max])

    raise ValueError(
        f"Invalid spatial_ref length: expected 2 (point) or 4 (bbox); got {len(vals)} for ref: {ref}."
    )

def _polygon_mask(points_px: List[Tuple[int, int]], W: int, H: int) -> np.ndarray:
    """Rasterize a polygon (in pixel coords) into a 0/1 mask of shape (H, W)."""
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)
    try:
        draw.polygon(points_px, outline=1, fill=1)
    except Exception as e:
        print(f"Moondream: polygon rasterization failed: {e}")
        return np.zeros((H, W), dtype=np.uint8)
    return np.array(mask_img, dtype=np.uint8)


def _mask_from_path_svg(
    path_val: Any,
    W: int,
    H: int,
    bbox_norm: Tuple[float, float, float, float] | None = None,
) -> np.ndarray | None:
    """Build mask from an SVG path string using svgpathtools.
    Moondream segment paths are normalized to [0,1] relative to the bbox.
    We therefore map each sampled path point into full-image coordinates via bbox.
    """

    path_obj = parse_path(str(path_val))
    if len(path_obj) == 0:
        return None

    if bbox_norm is not None:
        y_min, x_min, y_max, x_max = bbox_norm
        x_min = max(0.0, min(1.0, float(x_min)))
        y_min = max(0.0, min(1.0, float(y_min)))
        x_max = max(0.0, min(1.0, float(x_max)))
        y_max = max(0.0, min(1.0, float(y_max)))
        if x_max <= x_min or y_max <= y_min:
            return None
    else:
        x_min, y_min, x_max, y_max = 0.0, 0.0, 1.0, 1.0

    bw = x_max - x_min
    bh = y_max - y_min
    merged = np.zeros((H, W), dtype=np.uint8)
    points_px: List[Tuple[int, int]] = []
    prev_end = None

    def _flush_polygon(poly_pts: List[Tuple[int, int]]) -> None:
        nonlocal merged
        if len(poly_pts) < 3:
            return
        merged = np.maximum(merged, _polygon_mask(poly_pts, W, H))

    try:
        for seg in path_obj:
            if prev_end is not None and abs(seg.start - prev_end) > 1e-9:
                _flush_polygon(points_px)
                points_px = []

            # 64 samples per segment for reasonable fidelity
            for t in np.linspace(0.0, 1.0, 64, dtype=float):
                if t == 0.0 and points_px:
                    continue
                z = seg.point(t)
                local_x = max(0.0, min(1.0, float(z.real)))
                local_y = max(0.0, min(1.0, float(z.imag)))
                x_img = x_min + local_x * bw
                y_img = y_min + local_y * bh
                px = max(0, min(W - 1, int(round(x_img * (W - 1)))))
                py = max(0, min(H - 1, int(round(y_img * (H - 1)))))
                points_px.append((px, py))

            prev_end = seg.end
    except Exception as e:
        print(f"Moondream: SVG path sampling failed: {e}")
        return None

    _flush_polygon(points_px)
    if not np.any(merged):
        return None
    return merged


def _mask_from_path(
    path_val: Any,
    W: int,
    H: int,
    bbox_norm: Tuple[float, float, float, float] | None = None,
) -> np.ndarray | None:
    """High-level path-to-mask builder using only SVG parsing."""
    return _mask_from_path_svg(path_val, W, H, bbox_norm=bbox_norm)


class MoondreamVLMProvider(BBoxProvider):
    """Moondream VLM provider for both bounding boxes and segmentation."""

    def __init__(self):
        self.api_key = os.environ.get("MOONDREAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MOONDREAM_API_KEY must be provided via environment variable for MoondreamVLMProvider."
            )
        self._model = None

    def _model_lazy(self):
        if self._model is None:
            import moondream as md  # conditional import
            self._model = md.vl(api_key=self.api_key)
        return self._model

    # ----------------------------
    # Bounding boxes
    # ----------------------------
    def detect(self, image: Image.Image, object_prompt: str) -> list[dict]:
        model = self._model_lazy()
        result = model.detect(image, object_prompt)

        # Expected format:
        # { "objects": [ {"x_min":..., "y_min":..., "x_max":..., "y_max":...}, ... ] }
        objects = result.get("objects", []) if isinstance(result, dict) else []
        width, height = image.size
        detections = []
        for obj in objects:
            try:
                x_min = float(obj.get("x_min"))
                y_min = float(obj.get("y_min"))
                x_max = float(obj.get("x_max"))
                y_max = float(obj.get("y_max"))
            except (TypeError, ValueError):
                continue
            x1 = int(x_min * width)
            y1 = int(y_min * height)
            x2 = int(x_max * width)
            y2 = int(y_max * height)
            detections.append(
                {
                    "label": object_prompt,
                    "box_norm": [y_min, x_min, y_max, x_max],
                    "box_pixels": [x1, y1, x2, y2],
                }
            )
        return detections, result

    # ----------------------------
    # Pointing
    # ----------------------------
    def point(self, image_or_path: Any, prompt: str) -> Dict[str, List[Dict[str, float]]]:
        image = _ensure_pil(image_or_path)
        model = self._model_lazy()
        print(f"Moondream: point() prompt='{prompt}'")
        result = model.point(image, prompt)
        if not isinstance(result, dict):
            return {"points": []}
        pts = result.get("points") or []
        norm_pts = []
        for p in pts:
            try:
                norm_pts.append({"x": float(p["x"]), "y": float(p["y"])})
            except Exception:
                continue
        return {"points": norm_pts}
    
    # ----------------------------
    # Segmentation (new API)
    # ----------------------------
    def segment(self, image_or_path: Any, prompt_classes: List[str], spatial_refs: List[List[float]] = None) -> Any:
        """
        Segment each class in `prompt_classes` using Moondream.

        - If `spatial_refs` is provided, it is used directly as Moondream's
          `spatial_refs` argument for every prompt. Each ref can be either:
            [x, y]              normalized point (0..1), or pixel point (>1); or
            [x1, y1, x2, y2]    normalized box, or pixel box (>1 values).
          Pixel refs are normalized to 0..1 using image size.
        - If `spatial_refs` is None (default), falls back to the existing
          behavior: call `.point()` to get candidates and then call `.segment()`
          per point.
        """
        image = _ensure_pil(image_or_path)
        W, H = image.size
        model = self._model_lazy()

        all_predictions: List[Dict[str, Any]] = []
        all_points: Dict[str, List[Dict[str, float]]] = {}
        for prompt in (prompt_classes or []):
            refs_for_prompt = []

            if spatial_refs:
                norm_refs = [ _normalize_ref(ref, W, H) for ref in spatial_refs ]
                all_points[prompt] = [
                    {"x": v[0], "y": v[1]} for (kind, v) in norm_refs if kind == "point"
                ]
                refs_for_prompt = norm_refs
            else:
                try:
                    point_result = model.point(image, prompt)
                    pts = point_result.get("points", []) if isinstance(point_result, dict) else []
                except Exception as e:
                    print(f"Moondream: point() failed for prompt='{prompt}': {e}")
                    pts = []

                pts_norm: List[Dict[str, float]] = []
                for p in pts:
                    try:
                        pts_norm.append({"x": float(p["x"]), "y": float(p["y"])})
                    except Exception:
                        continue
                all_points[prompt] = pts_norm
                refs_for_prompt = [("point", [p["x"], p["y"]]) for p in pts_norm]

            if not refs_for_prompt:
                continue

            for (kind, v) in refs_for_prompt:
                if kind == "point":
                    sr = [v]
                else:
                    x_min, y_min, x_max, y_max = v
                    sr = [[x_min, y_min, x_max, y_max]]

                try:
                    seg_res = model.segment(image, prompt, spatial_refs=sr)
                except Exception as e:
                    print(f"Moondream: segment() failed for prompt='{prompt}' with spatial_ref={sr}: {e}")
                    continue

                path_val = None
                bbox_norm = None
                if isinstance(seg_res, dict):
                    path_val = seg_res.get("path")
                    bbox = seg_res.get("bbox") or {}
                    try:
                        x_min = float(bbox.get("x_min"))
                        y_min = float(bbox.get("y_min"))
                        x_max = float(bbox.get("x_max"))
                        y_max = float(bbox.get("y_max"))
                        bbox_norm = (y_min, x_min, y_max, x_max)
                    except Exception:
                        bbox_norm = None

                if bbox_norm is None:
                    if kind == "point":
                        px, py = v
                        eps = 2.0 / max(1.0, float(W))
                        bbox_norm = (
                            max(0.0, py - eps),
                            max(0.0, px - eps),
                            min(1.0, py + eps),
                            min(1.0, px + eps),
                        )
                    else:
                        y_min = v[1]; x_min = v[0]; y_max = v[3]; x_max = v[2]
                        bbox_norm = (y_min, x_min, y_max, x_max)

                # Build seg_mask from SVG path in bbox-relative coordinates.
                seg_mask = None
                try:
                    if path_val is not None:
                        seg_mask = _mask_from_path(path_val, W, H, bbox_norm=bbox_norm)
                except Exception as e:
                    print(f"Moondream: seg_mask from path failed: {e}")
                    seg_mask = None

                y_min, x_min, y_max, x_max = bbox_norm
                x1 = max(0, min(W, int(round(x_min * W))))
                y1 = max(0, min(H, int(round(y_min * H))))
                x2 = max(0, min(W, int(round(x_max * W))))
                y2 = max(0, min(H, int(round(y_max * H))))
                
                all_predictions.append(
                    {
                        "class": str(prompt),
                        "bbox_norm": [y_min, x_min, y_max, x_max],
                        "bbox_pixels": [x1, y1, x2, y2],
                        "path": path_val,
                        "seg_mask": seg_mask,
                    }
                )
        return {"predictions": all_predictions, "points": all_points}




