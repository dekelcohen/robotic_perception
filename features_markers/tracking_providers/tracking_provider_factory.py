from .tracking_providers import TrackingProvider

def get_tracking_provider(name: str, initial_bbox_pixels: list[int]) -> TrackingProvider:
    name = (name or "").lower()
    if name == "csrt":
        from .csrt_tracker_provider import CSRTTrackerProvider
        return CSRTTrackerProvider(initial_bbox_pixels)
    raise ValueError(f"Unknown tracker: {name}")
