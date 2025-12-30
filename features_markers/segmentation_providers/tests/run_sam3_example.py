from __future__ import annotations

"""
Minimal smoke test for the SAM3 segmentation provider using the
paths and prompts provided by the user.

Usage:
    python -m features_markers.segmentation_providers.tests.run_sam3_example

Requirements:
    pip install inference-sdk pycocotools
    set ROBOFLOW_API_KEY=...  (Windows PowerShell: $env:ROBOFLOW_API_KEY="...")
"""

from typing import List
from PIL import Image

from features_markers.segmentation_providers.roboflow_helper import (
    create_roboflow_client,
)
from features_markers.segmentation_providers.sam3_segmentation_provider import (
    SAM3SegmentationProvider,
)


def main():
    # Provided sample path and prompts
    path_to_jpg = r"D:\Docs\test6\Projects\Robotics\Samples\Moley_Robot_Kitchen.jpg"
    prompt_classes: List[str] = ["golden pot handles"]

    # Create client via helper and pass into provider
    client = create_roboflow_client()
    provider = SAM3SegmentationProvider(client=client)

    # Load the image and pass the in-memory object to the provider
    image = Image.open(path_to_jpg).convert("RGB")
    results = provider.segment(image, prompt_classes=prompt_classes)
    print(results)


if __name__ == "__main__":
    main()
