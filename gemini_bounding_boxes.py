# Demonstration of using Gemini API to detect objects and draw bounding boxes on an image.

# conda activate env_woderfullteam
  ## pip install google-genai opencv-python
# cd /d D:\NLP\Robotics\robotic_perception
# cd /d E:\Robotics\Robotics VLM\robotic_perception
# command line:
# --input "D:\Docs\test6\Projects\Robotics\Samples\Moley_Robot_Kitchen.jpg" --output ./outputs/Moley_Robot_Kitchen.jpg --prompt "Detect the closest  golden handle of the large pot (from obs pov) and return bounding boxes as [ymin, xmin, ymax, xmax] followed by label."



##### ============================= Rest service Gemini =====================================================================

# gemini_rest_bbox.py
# REST-based demo using Gemini API key to detect objects and draw bounding boxes.

import os
import argparse
import base64
import json
import re
import cv2
import numpy as np
import requests
from PIL import ImageDraw, Image, ImageColor
from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# Load environment variables
# ----------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in environment.")

# ----------------------------------------------------------------------------
# Function: call_gemini_rest
# Sends a POST request and returns full response JSON
# ----------------------------------------------------------------------------
def call_gemini_rest(image_path: str, prompt: str) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                {"text": prompt}
            ]
        }]
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
    )
    resp = requests.post(url, params={"key": API_KEY}, json=payload)
    resp.raise_for_status()
    return resp.json()

# ----------------------------------------------------------------------------
# Function: extract_json_from_text
# Finds a ```json ... ``` block in the given markdown text
# ----------------------------------------------------------------------------
def extract_json_from_text(md: str) -> list:
    pattern = r"```json\s*(\[.*?\])\s*```"
    match = re.search(pattern, md, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in model output")
    return json.loads(match.group(1))

# ----------------------------------------------------------------------------
# Function: draw_boxes
# Draws boxes on the image and saves output
# ----------------------------------------------------------------------------
def draw_boxes(image_path, boxes, output_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    colors = list(ImageColor.colormap.values())
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for i, item in enumerate(boxes):
        # Handle different item formats
        if isinstance(item, dict):
            coords = item.get("box_2d")
            label = item.get("label", "")
        elif isinstance(item, list) and len(item) >= 2:
            coords, label = item[0], item[1]
        else:
            continue
        # Draw if coords valid
        if not coords or len(coords) != 4:
            continue
        ymin, xmin, ymax, xmax = coords
        x1, y1 = int(xmin/1000*w), int(ymin/1000*h)
        x2, y2 = int(xmax/1000*w), int(ymax/1000*h)
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if label:
            draw.text((x1+4, y1+4), str(label), fill=color)

    out_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, out_img)

# ----------------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini REST Bounding Box Demo")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save annotated image")
    parser.add_argument("--prompt", default="Return bounding boxes as [ymin, xmin, ymax, xmax] label", help="Detection prompt")
    args = parser.parse_args()

    # Call REST API
    resp = call_gemini_rest(args.input, args.prompt)
    # Extract model's text markdown
    md = resp["candidates"][0]["content"]["parts"][0]["text"]
    print('model response (text):',md)
    # md = '```json[{"box_2d": [384, 755, 663, 1000], "label": "the golden sink"}]```'
    # Extract JSON array
    boxes = extract_json_from_text(md)
    # Draw and save
    draw_boxes(args.input, boxes, args.output)
    print(f"Annotated image saved to {args.output}")

