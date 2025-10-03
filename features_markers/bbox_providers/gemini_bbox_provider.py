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
from .bbox_providers import BBoxProvider

class GeminiBBoxProvider(BBoxProvider):
    """Bounding box provider using the Gemini API."""
    
    GEMINI_MODEL = "gemini-2.5-flash"
    GEMINI_REST_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided via argument or environment variable for GeminiBBoxProvider.")

    def _gemini_rest_call_image_and_text(self, image_bytes: bytes, mime_type: str, prompt_text: str, timeout: int = 30):
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content_part = {"inline_data": {"mime_type": mime_type, "data": b64}}
        payload = {"contents": [{"parts": [content_part, {"text": prompt_text}]}]}
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        resp = requests.post(self.GEMINI_REST_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return self._extract_text_from_gemini_response(data), data

    def _extract_text_from_gemini_response(self, resp_json: dict) -> str:
        try:
            cands = resp_json.get("candidates") or []
            if cands:
                cand0 = cands[0]
                content = cand0.get("content") or cand0.get("message") or {}
                parts = content.get("parts") if isinstance(content, dict) else None
                if parts:
                    texts = []
                    for p in parts:
                        if isinstance(p, dict):
                            if "text" in p and p["text"]:
                                texts.append(p["text"])
                            elif "html" in p and p["html"]:
                                texts.append(p["html"])
                    if texts:
                        return "\n".join(texts)
        except Exception:
            pass

        def find_texts(obj):
            found = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("text", "html") and isinstance(v, str):
                        found.append(v)
                    else:
                        found += find_texts(v)
            elif isinstance(obj, list):
                for item in obj:
                    found += find_texts(item)
            return found

        found_texts = find_texts(resp_json)
        if found_texts:
            return "\n".join(found_texts)
        return json.dumps(resp_json)

    def _extract_first_json_like(self, text: str):
        text = text.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except Exception:
                pass
        patterns = [r"(\{.*\})", r"(\[.*\])"]
        for pat in patterns:
            m = re.search(pat, text, flags=re.DOTALL)
            if m:
                candidate = m.group(1)
                try:
                    return json.loads(candidate)
                except Exception:
                    for end in range(len(candidate), 0, -1):
                        try:
                            return json.loads(candidate[:end])
                        except Exception:
                            continue
        raise ValueError("No JSON found in text.")

    def detect(self, image: Image.Image, object_prompt: str) -> list[dict]:
        img_buf = BytesIO()
        image.save(img_buf, format="JPEG")
        img_bytes = img_buf.getvalue()
        mime_type = "image/jpeg"

        bbox_prompt = (
            f"Return a JSON array of objects detected corresponding only to the object '{object_prompt}' in the image. "
            "For each object return a dictionary with keys: 'label' (string) and 'box_2d' (list of integers [ymin, xmin, ymax, xmax]) "
            "where coordinates are normalized to the range 0..1000 (y then x). "
            "Respond with only valid JSON (an array) and no additional text."
        )
        print("Requesting bounding boxes from Gemini (image + prompt)...")
        try:
            gem_text, raw_resp = self._gemini_rest_call_image_and_text(img_bytes, mime_type, bbox_prompt)
        except Exception as e:
            print("Error calling Gemini for bounding boxes:", e)
            raise

        print("Raw Gemini text response (truncated):")
        print(gem_text[:1000])

        try:
            parsed = self._extract_first_json_like(gem_text)
        except Exception as e:
            print("Could not parse JSON from Gemini response. Raw text:")
            print(gem_text)
            raise

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise RuntimeError("Unexpected parsed JSON from Gemini: not list or dict.")

        width, height = image.size
        detections = []
        for item in parsed:
            label = item.get("label") or item.get("name") or object_prompt
            box = item.get("box_2d") or item.get("box") or item.get("box2d")
            if not box or not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            ymin, xmin, ymax, xmax = [float(x) for x in box[:4]]
            x1 = int(xmin / 1000.0 * width)
            y1 = int(ymin / 1000.0 * height)
            x2 = int(xmax / 1000.0 * width)
            y2 = int(ymax / 1000.0 * height)
            detections.append({
                "label": label,
                "box_norm": [ymin, xmin, ymax, xmax],
                "box_pixels": [x1, y1, x2, y2],
            })
        return detections, parsed