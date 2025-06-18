# conda activate env_woderfullteam
# cd /d D:\NLP\Robotics\robotic_perception
# gemini_bbox_demo.py
# Demonstration of using Gemini API to detect objects and draw bounding boxes on an image.

import os
import io
import argparse
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from google import genai
from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# Load environment variables from .env file
# ----------------------------------------------------------------------------
load_dotenv()

# ----------------------------------------------------------------------------
# Function: init_client
# Initializes the Gemini API client using an API key from environment variable.
# Returns:
#   genai.Client instance
# ----------------------------------------------------------------------------
def init_client(api_key_env: str = "GEMINI_API_KEY") -> genai.Client:
    """
    Initialize and return a Gemini API client.

    Args:
        api_key_env: Name of the environment variable holding your API key.

    Returns:
        genai.Client configured with your API key.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable '{api_key_env}' not set. Please add it to your .env file.")
    return genai.Client(api_key=api_key)

# ----------------------------------------------------------------------------
# Function: load_image_bytes
# Loads an image from a file path into raw bytes.
# Returns:
#   BytesIO containing the image data.
# ----------------------------------------------------------------------------
def load_image_bytes(image_path: str) -> io.BytesIO:
    """
    Read an image file and return an in-memory bytes buffer.

    Args:
        image_path: Path to the image file.

    Returns:
        io.BytesIO containing the image data.
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return io.BytesIO(img_bytes)

# ----------------------------------------------------------------------------
# Function: detect_bounding_boxes
# Sends the image and prompt to Gemini API and retrieves bounding box response.
# Returns:
#   Raw response from the API.
# ----------------------------------------------------------------------------
def detect_bounding_boxes(
    client: genai.Client,
    image_input: Union[io.BytesIO, str],
    prompt: str
) -> List[str]:
    """
    Call Gemini generate_content to detect objects and bounding boxes.

    Args:
        client: Initialized Gemini API client.
        image_input: Either a BytesIO buffer or a file reference string.
        prompt: Instruction prompt asking for bounding boxes.

    Returns:
        List of raw string lines with bounding box data.
    """
    inputs = [image_input, prompt]
    response = client.models.generate_content(inputs)
    # Assume response.content is a single string with newline-separated entries
    return response.content.splitlines()

# ----------------------------------------------------------------------------
# Function: parse_boxes
# Parses raw API lines of the form:
#   "[ymin, xmin, ymax, xmax] label"
# into structured data.
# Returns:
#   List of (box, label)
# ----------------------------------------------------------------------------
def parse_boxes(raw_lines: List[str]) -> List[Tuple[List[float], str]]:
    """
    Convert API string lines into numeric bounding boxes and labels.

    Args:
        raw_lines: List of strings from the API response.

    Returns:
        List of tuples (box_coords, label), where box_coords is [ymin, xmin, ymax, xmax].
    """
    parsed = []
    for line in raw_lines:
        try:
            # Remove brackets and split
            coords_str, label = line.strip().split(']')
            coords = coords_str.lstrip('[').split(',')
            box = [float(c) for c in coords]
            parsed.append((box, label.strip()))
        except Exception:
            # Skip unparseable lines
            continue
    return parsed

# ----------------------------------------------------------------------------
# Function: normalize_to_pixels
# Converts a normalized 0-1000 box to pixel coordinates for a given image size.
# Returns:
#   Tuple of (x1, y1, x2, y2)
# ----------------------------------------------------------------------------
def normalize_to_pixels(
    box: List[float],
    image_size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Scale a [ymin, xmin, ymax, xmax] box from 0-1000 space to pixels.

    Args:
        box: Normalized coordinates [ymin, xmin, ymax, xmax].
        image_size: (width, height) of the image in pixels.

    Returns:
        (x1, y1, x2, y2) pixel coordinates.
    """
    ymin, xmin, ymax, xmax = box
    width, height = image_size
    x1 = int(xmin / 1000 * width)
    y1 = int(ymin / 1000 * height)
    x2 = int(xmax / 1000 * width)
    y2 = int(ymax / 1000 * height)
    return x1, y1, x2, y2

# ----------------------------------------------------------------------------
# Function: draw_boxes
# Draws bounding boxes and labels on an image and saves the output.
# ----------------------------------------------------------------------------
def draw_boxes(
    input_path: str,
    boxes_labels: List[Tuple[List[float], str]],
    output_path: str
) -> None:
    """
    Open an image, draw bounding boxes with labels, and save to disk.

    Args:
        input_path: Path to the source image file.
        boxes_labels: List of tuples ([ymin, xmin, ymax, xmax], label).
        output_path: Path to save the annotated image.
    """
    # Load image with OpenCV
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Unable to load image '{input_path}'")

    height, width = img.shape[:2]
    
    for box, label in boxes_labels:
        x1, y1, x2, y2 = normalize_to_pixels(box, (width, height))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    cv2.imwrite(output_path, img)

# ----------------------------------------------------------------------------
# Main Execution
# Parses arguments, runs detection, and outputs the result image.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini API Bounding Box Demo")
    parser.add_argument("--input", required=True, help="Path to input image file.")
    parser.add_argument("--output", required=True, help="Path to save annotated image.")
    parser.add_argument(
        "--prompt",
        default="Detect all objects and return bounding boxes as [ymin, xmin, ymax, xmax] followed by label.",
        help="Prompt for Gemini API."
    )
    args = parser.parse_args()

    # Initialize client and load image bytes
    client = init_client()
    img_buf = load_image_bytes(args.input)

    # Detect boxes
    raw = detect_bounding_boxes(client, img_buf, args.prompt)
    parsed = parse_boxes(raw)

    # Draw and save
    draw_boxes(args.input, parsed, args.output)

    print(f"Annotated image saved to {args.output}")

# ----------------------------------------------------------------------------
# .env file example:
# GEMINI_API_KEY=your_api_key_here
# ----------------------------------------------------------------------------
