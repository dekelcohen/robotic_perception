"""
Segmentation interface and types.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Any


class ISegmentation(ABC):
    """Abstract segmentation provider interface."""

    @abstractmethod
    def segment(self, image_or_path: Any, prompt_classes: List[str]) -> Any:
        """
        Segment an image using text prompts and return provider-native results.

        Args:
            image_or_path: Either a path to an image on disk (str) or an in-memory image
                (e.g., PIL.Image.Image, bytes). Providers should handle both.
            prompt_classes: List of class prompts to segment (provider-specific semantics).

        Returns:
            Provider-specific result object. For SAM3, returns a list of items where
            each item contains model outputs and each prediction may include an added
            `np_mask` key with a decoded numpy binary mask.
        """
        raise NotImplementedError
