"""
Thin wrapper for creating a RoboFlow inference client and helpers
for decoding RLE masks to NumPy arrays.
"""
from __future__ import annotations

from typing import Any, List
import os
import numpy as np


def create_roboflow_client(api_url: str = "https://serverless.roboflow.com", api_key: str | None = None):
    """
    Create and return a RoboFlow InferenceHTTPClient.

    If api_key is None, InferenceHTTPClient will look it up from env if supported.
    """
    try:
        from inference_sdk import InferenceHTTPClient
    except Exception as e:
        # Keep error visible; caller can decide how to handle.
        raise RuntimeError(
            "Missing dependency inference-sdk. Please `pip install inference-sdk pycocotools`."
        ) from e

    # Resolve API key from env if not provided explicitly
    if api_key is None:
        api_key = (
            os.environ.get("ROBOFLOW_API_KEY")
            or os.environ.get("ROBOFLOW_APIKEY")           
        )
    if not api_key:
        raise ValueError(
            "RoboFlow API key is missing. Set ROBOFLOW_API_KEY env var or pass api_key."
        )

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    return client


def add_np_masks_to_results(results: List[dict]) -> List[dict]:
    """
    Iterate through workflow results and add a decoded NumPy mask (H,W uint8)
    as `np_mask` next to each prediction's `rle_mask` if present.
    """
    try:
        from pycocotools import mask as mask_utils
    except Exception as e:
        raise RuntimeError(
            "Missing dependency pycocotools. Please `pip install pycocotools`."
        ) from e

    for item in results or []:
        for _model_key, model_output in (item.items() if isinstance(item, dict) else []):
            predictions = model_output.get("predictions", []) if isinstance(model_output, dict) else []
            for pred in predictions:
                rle = pred.get("rle_mask") if isinstance(pred, dict) else None
                if not rle:
                    continue
                decoded = mask_utils.decode(rle)
                if decoded is None:
                    continue
                if decoded.ndim == 3:
                    decoded = decoded[:, :, 0]
                pred["np_mask"] = (decoded > 0).astype(np.uint8)
    return results
