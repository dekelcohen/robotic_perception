from PIL import Image
import os
import json
# ...existing imports if needed...
from .bbox_providers import BBoxProvider

class MoondreamBBoxProvider(BBoxProvider):
    """Bounding box provider using the Moondream VLM API."""
    
    def __init__(self):
        self.api_key = os.environ.get("MOONDREAM_API_KEY")
        if not self.api_key:
            raise ValueError("MOONDREAM_API_KEY must be provided via environment variable for MoondreamBBoxProvider.")
    
    def detect(self, image: Image.Image, object_prompt: str) -> list[dict]:
        # Conditional import to limit dependencies
        import moondream as md
        
        # Initialize the moondream model with the API key
        model = md.vl(api_key=self.api_key)
        
        # Call the detect method of the moondream model
        result = model.detect(image, object_prompt)
        
        # Updated VLM response parsing example:
        # {
        #   "objects": [
        #     {
        #       "x_min": 0.3914136774837971,
        #       "y_min": 0.4302837625145912,
        #       "x_max": 0.4933519475162029,
        #       "y_max": 0.6165912374854088
        #     }
        #   ]
        # }
        objects = result.get("objects", [])
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
            detections.append({
                "label": object_prompt,  # no label provided in the response
                "box_norm": [y_min, x_min, y_max, x_max],
                "box_pixels": [x1, y1, x2, y2],
            })
        return detections, result
