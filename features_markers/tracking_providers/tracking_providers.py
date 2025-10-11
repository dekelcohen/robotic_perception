from typing import Tuple
from PIL import Image

class TrackingProvider:
    """Base class for tracking providers."""
    def __init__(self):
        pass
    
    def update(self, image: Image.Image) -> Tuple[bool,list[int]]:
        """
        Update the tracker with a new frame and return the current bbox [x1, y1, x2, y2].
        Return success (bool), bbox. [x1,y,x2,y2]
        If failed return False, with special [0,1,0,1] bbox - so models can be trained to ignore it when failed and use other features (bboxes from other cams ..)
        """
        raise NotImplementedError
