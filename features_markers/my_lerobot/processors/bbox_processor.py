#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import json

import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE

# Imports from the user's robotic_perception library.
# Please ensure this library is in your PYTHONPATH.
from bbox_providers.bbox_provider_factory import get_bbox_provider
from convert_lerobot_dataset_to_bbox import (
    FrameProcessor,
    FrameProcessorOptions,
    tensor_to_pil,
)
from tracking_providers.tracking_provider_factory import (
    get_tracking_provider,
)

from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="bbox_processor")
class BboxProcessorStep(ObservationProcessorStep):
    """
    A processor step that uses a FrameProcessor to add bounding box information to the observation.

    This step extracts a specific camera image from the observation, passes it to a
    FrameProcessor to get a bounding box for a target object, and adds the bounding box
    to the observation under the key "observation.environment_state".

    It can optionally replace the original camera image with an annotated version showing
    the detected bounding box.

    Args:
        camera_key: The key for the camera image in the observation dictionary.
        object_prompt: A text description of the object to detect.
        bbox_detector: The name of the bounding box detector to use (e.g., "gemini", "moondream").
        tracker: The name of the tracker to use for propagating the bounding box across frames.
                 Defaults to "none".
        annotate_image: If True, the camera image is replaced with an annotated version.
                        Defaults to False.
    """

    camera_key: str
    object_prompt: str
    bbox_detector: str
    tracker: str = "none"
    annotate_image: bool = False

    def __post_init__(self):
        """Initializes the FrameProcessor with the specified providers."""
        bbox_provider = get_bbox_provider(self.bbox_detector)
        tracker_provider = None
        if self.tracker and self.tracker.lower() != "none":
            tracker_provider = get_tracking_provider(self.tracker)
        options = FrameProcessorOptions(max_tracker_failed_saves=1) # save one image when tracking fails during inference (episode=-1,frame=-1)
        self.frame_processor = FrameProcessor(self.object_prompt, bbox_provider, tracker_provider, options)
        self.last_bbox = None

    @classmethod
    def from_cfg_dict(cls, cfg: dict) -> "BboxProcessorStep":
        """Instantiates a BboxProcessorStep from a JSON configuration."""
        return cls(
            camera_key=cfg["camera_key"],
            object_prompt=cfg["object_prompt"],
            bbox_detector=cfg["bbox_detector"],
            tracker=cfg.get("tracker", "none"),
            annotate_image=cfg.get("annotate_image", False),
        )
        
    @classmethod
    def from_json(cls, bbox_config_path: str) -> "BboxProcessorStep":
        """Instantiates a BboxProcessorStep from a JSON configuration."""
        with open(bbox_config_path, "r") as f:
            dct_cfg = json.load(f)
            return cls.from_cfg_dict(dct_cfg)

    def observation(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Processes the observation to add bounding box data.
        """
        if self.camera_key not in observation:
            raise ValueError(f"Camera key '{self.camera_key}' not found in observation.")

        # The observation processor expects batched inputs, but FrameProcessor works on single images.
        # We process the first image in the batch.
        # Squeeze batch dimension if present, assuming it's the first one.
        frame_tensor = observation[self.camera_key]
        if frame_tensor.ndim == 4:
            frame_tensor = frame_tensor[0]

        pil_img = tensor_to_pil(frame_tensor)

        annotated_frame, new_bbox = self.frame_processor.process(
            pil_img, bbox=self.last_bbox, annotate=self.annotate_image
        )

        if new_bbox:
            self.last_bbox = new_bbox

        if self.annotate_image:
            # Replace the original image with the annotated one
            annotated_tensor = torch.from_numpy(np.array(annotated_frame)).permute(2, 0, 1).float() / 255.0
            # Add back the batch dimension
            observation[self.camera_key] = annotated_tensor.unsqueeze(0)

        if self.last_bbox:
            bbox_tensor = torch.tensor(self.last_bbox, dtype=torch.float32)
        else:
            # If no bbox is found, return a dummy bbox of zeros
            bbox_tensor = torch.zeros(4, dtype=torch.float32)

        # Add back the batch dimension for the environment state
        observation[OBS_ENV_STATE] = bbox_tensor.unsqueeze(0)
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Adds the bounding box feature to the policy's observation features.
        """
        # Ensure the observation feature type exists
        if "observation" not in features:
            features["observation"] = {}

        features["observation"][OBS_ENV_STATE] = {
            "shape": (4,),
            "dtype": "float32",
            "name": ["y0", "x0", "y1", "x1"],
        }
        return features
