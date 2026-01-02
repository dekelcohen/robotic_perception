from __future__ import annotations

from typing import Any, Dict, List, Tuple
from PIL import Image
import os
import json

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
    def segment(self, image_or_path: Any, prompt_classes: List[str]) -> Any:
        image = _ensure_pil(image_or_path)
        W, H = image.size
        model = self._model_lazy()

        all_predictions: List[Dict[str, Any]] = []
        all_points: Dict[str, List[Dict[str, float]]] = {}

        for prompt in (prompt_classes or []):
            # 1) points
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
            if not pts_norm:
                continue

            # 2) segment per point
            for p in pts_norm:
                try:
                    seg_res = model.segment(image, prompt, spatial_refs=[[p["x"], p["y"]]])
                except Exception as e:
                    print(f"Moondream: segment() failed for prompt='{prompt}' at point={p}: {e}")
                    continue

                path_str = None
                bbox_norm = None
                if isinstance(seg_res, dict):
                    path_str = seg_res.get("path")
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
                    px, py = float(p["x"]), float(p["y"])\
                    
                    
                    eps = 2.0 / max(1.0, float(W))
                    bbox_norm = (
                        max(0.0, py - eps),
                        max(0.0, px - eps),
                        min(1.0, py + eps),
                        min(1.0, px + eps),
                    )

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
                        "path": path_str,
                    }
                )

        return {"predictions": all_predictions, "points": all_points}
