from PIL import Image
import numpy as np
from .tracking_providers import TrackingProvider

class CSRTTrackerProvider(TrackingProvider):
    """OpenCV CSRT tracker provider (requires opencv-contrib-python)."""

    def __init__(self):
        super().__init__()
        self._cv2 = None
        self._tracker = None
        self._initialized = False

    def _ensure_cv2(self):
        if self._cv2 is None:
            import cv2  # lazy import
            self._cv2 = cv2

    def _create_tracker(self):
        self._ensure_cv2()
        cv2 = self._cv2
        # Try modern API, then legacy fallback
        tracker = None
        if hasattr(cv2, "TrackerCSRT_create"):
            tracker = cv2.TrackerCSRT_create()
        elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            tracker = cv2.legacy.TrackerCSRT_create()
        else:
            raise RuntimeError("OpenCV CSRT tracker not available. Install opencv-contrib-python.")
        return tracker

    def _pil_to_bgr(self, image: Image.Image):
        self._ensure_cv2()
        cv2 = self._cv2
        arr = np.array(image)
        if arr.ndim == 2:
            return arr
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def update(self, image: Image.Image) -> list[int]:        
        cv2 = None        
        if not self._initialized:
            self._tracker = self._create_tracker()
            cv2 = self._cv2
            frame = self._pil_to_bgr(image)
            x1, y1, x2, y2 = self.initial_bbox_pixels
            w = max(1, int(x2 - x1))
            h = max(1, int(y2 - y1))
            init_ok = self._tracker.init(frame, (int(x1), int(y1), w, h))
            self._initialized = True
            #if not init_ok: the tracker.init sometimes return None - so cannot know if succeeded 
                # If init fails, return initial bbox
            return True, [int(x1), int(y1), int(x2), int(y2)]

        if cv2 is None:
            cv2 = self._cv2
        frame = self._pil_to_bgr(image)
        ok, bb = self._tracker.update(frame)
        if not ok:
            # If tracking fails, return a dummy bbox 
            return ok, None # Special sent for failure to track - policies should be trained with this 

        x, y, w, h = bb
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))

        # Clamp to frame bounds
        H, W = frame.shape[:2]
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        if x2 <= x1: x2 = min(W - 1, x1 + 1)
        if y2 <= y1: y2 = min(H - 1, y1 + 1)
        self.initial_bbox_pixels = [x1, y1, x2, y2]  # update last known
        return ok, [x1, y1, x2, y2]
