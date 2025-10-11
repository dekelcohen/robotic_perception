from .tracking_providers import TrackingProvider

def get_tracking_provider(name: str) -> TrackingProvider:
    name = (name or "").lower()
    if name == "csrt":
        from .csrt_tracker_provider import CSRTTrackerProvider
        return CSRTTrackerProvider()
    raise ValueError(f"Unknown tracker: {name}")
