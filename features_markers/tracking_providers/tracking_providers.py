from PIL import Image

class TrackingProvider:
    """Base class for tracking providers."""
    def __init__(self, initial_bbox_pixels: list[int]):
        # [x1, y1, x2, y2]
        self.initial_bbox_pixels = list(initial_bbox_pixels)

    def update(self, image: Image.Image) -> list[int]:
        """
        Update the tracker with a new frame and return the current bbox [x1, y1, x2, y2].
        """
        raise NotImplementedError
