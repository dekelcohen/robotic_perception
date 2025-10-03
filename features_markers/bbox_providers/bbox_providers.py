"""
Bounding box detection providers.
"""
import os
import base64
import json
import re
from io import BytesIO
import requests
from PIL import Image

class BBoxProvider:
    """Base class for bounding box providers."""
    def detect(self, image: "PIL.Image.Image", object_prompt: str) -> list[dict]:
        """
        Detects bounding boxes for a given object in an image.

        Args:
            image: The input PIL image.
            object_prompt: The text prompt describing the object to detect.

        Returns:
            A list of detections, where each detection is a dictionary with
            'label', 'box_norm' (normalized coordinates), and 'box_pixels'.
        """
        raise NotImplementedError