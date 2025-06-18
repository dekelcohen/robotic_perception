# Demonstration of using Gemini API to detect objects and draw bounding boxes on an image.

# conda activate env_woderfullteam
  ## pip install google-genai opencv-python
# cd /d D:\NLP\Robotics\robotic_perception
# cd /d E:\Robotics\Robotics VLM\robotic_perception
# command line:
# --input "D:\Docs\test6\Projects\Robotics\Samples\Moley_Robot_Kitchen.jpg" --output ./outputs/Moley_Robot_Kitchen.jpg --prompt "Detect the closest  golden handle of the large pot (from obs pov) and return bounding boxes as [ymin, xmin, ymax, xmax] followed by label."



# gemini_bbox_demo.py
# Demonstration of using Gemini API (GenAI SDK with Vertex AI) to detect objects
# and draw bounding boxes on an image, using the updated Google GenAI SDK API.

import os
import io
import argparse
from typing import List, Tuple

from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw
import requests
from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HttpOptions,
    Part,
    SafetySetting
)

# ----------------------------------------------------------------------------
# Load environment variables from .env file
# ----------------------------------------------------------------------------
load_dotenv()
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")  # enable Vertex AI backend

# ----------------------------------------------------------------------------
# Pydantic model for structured response schema
# ----------------------------------------------------------------------------
class BoundingBox(BaseModel):
    """
    Represents a bounding box with normalized coordinates and label.
    box_2d: [ymin, xmin, ymax, xmax] in 0-1000 scale
    label: object label
    """
    box_2d: List[int]
    label: str

# ----------------------------------------------------------------------------
# Function: init_client
# Initializes the GenAI SDK client with Vertex AI settings.
# ----------------------------------------------------------------------------
def init_client() -> genai.Client:
    """
    Initialize and return a Gemini (GenAI) client.
    Requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION in env.
    """
    #project = os.getenv("GOOGLE_CLOUD_PROJECT")
    #location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    #if not project:
    #    raise ValueError("Set GOOGLE_CLOUD_PROJECT in .env to your GCP project ID.")
    # HTTP options config for GenAI SDK
    #http_opts = HttpOptions(api_version="v1")
    #return genai.Client(http_options=http_opts)
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ----------------------------------------------------------------------------
# Function: generate_config
# Builds GenerateContentConfig with safety and schema.
# ----------------------------------------------------------------------------
def generate_config() -> GenerateContentConfig:
    """
    Return content configuration: system instructions, safety, JSON schema.
    """
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                "label": {"type": "string"}
            },
            "required": ["box_2d", "label"]
        }
    }
    return GenerateContentConfig(
        system_instruction="""
        Return bounding boxes as JSON array of {box_2d:[ymin,xmin,ymax,xmax], label}.
        Limit to 25 objects. Do not return masks.
        """,
        temperature=0.0,
        safety_settings=[
            SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH"
            )
        ],
        response_mime_type="application/json",
        response_schema=schema
    )

# ----------------------------------------------------------------------------
# Function: detect_bounding_boxes
# Calls the GenAI SDK to detect objects and return parsed BoundingBox objects.
# ----------------------------------------------------------------------------
def detect_bounding_boxes(
    client: genai.Client,
    image_path: str,
    prompt: str
) -> List[BoundingBox]:
    """
    Send image and prompt to GenAI SDK and parse JSON bounding boxes.

    Args:
        client: Initialized GenAI client
        image_path: Local path to image file
        prompt: Instruction for object detection

    Returns:
        List of BoundingBox instances
    """
    # Read bytes and wrap in Part
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        
    part = Part.from_bytes(
        data=img_bytes,
        mime_type="image/jpeg"
    )

    config = generate_config()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[part, prompt],
        config=config
    )
    # response.parsed is List[BoundingBox]
    return response.parsed  # type: ignore

# ----------------------------------------------------------------------------
# Function: plot_bounding_boxes
# Draws bounding boxes on image using PIL and shows or saves it.
# ----------------------------------------------------------------------------
def plot_bounding_boxes(
    image_path: str,
    boxes: List[BoundingBox],
    output_path: str = None
) -> None:
    """
    Draws colored boxes with labels on the image.

    Args:
        image_path: Local path or URL
        boxes: List of BoundingBox
        output_path: If set, saves annotated image
    """
    # Load image
    if image_path.startswith("http"):
        im = Image.open(requests.get(image_path, stream=True).raw)
    else:
        im = Image.open(image_path)
    width, height = im.size
    draw = ImageDraw.Draw(im)
    colors = list(ImageColor.colormap.keys())

    for idx, bb in enumerate(boxes):
        ymin, xmin, ymax, xmax = bb.box_2d
        x1 = int(xmin / 1000 * width)
        y1 = int(ymin / 1000 * height)
        x2 = int(xmax / 1000 * width)
        y2 = int(ymax / 1000 * height)
        color = colors[idx % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, y1 + 4), bb.label, fill=color)

    if output_path:
        im.save(output_path)
        print(f"Annotated image saved to {output_path}")
    else:
        im.show()


# ----------------------------------------------------------------------------
# Main Execution: CLI
# ----------------------------------------------------------------------------
if False and __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini GenAI Bounding Box Demo")
    parser.add_argument("--input", required=True, help="Path or URL to input image.")
    parser.add_argument("--output", help="Path to save annotated image.")
    parser.add_argument(
        "--prompt",
        default="Detect all objects and return bounding boxes as [ymin, xmin, ymax, xmax] with label.",
        help="Prompt for GenAI SDK."
    )
    args = parser.parse_args()

    client = init_client()
    boxes = detect_bounding_boxes(client, args.input, args.prompt)
    plot_bounding_boxes(args.input, boxes, args.output)

# ----------------------------------------------------------------------------
# .env file example:
# GOOGLE_CLOUD_PROJECT=your-project-id
# GOOGLE_CLOUD_LOCATION=global
# GEMINI_API_KEY=not-used-with-VertexAI
# ----------------------------------------------------------------------------


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
    # Extract JSON array
    boxes = extract_json_from_text(md)
    # Draw and save
    draw_boxes(args.input, boxes, args.output)
    print(f"Annotated image saved to {args.output}")

