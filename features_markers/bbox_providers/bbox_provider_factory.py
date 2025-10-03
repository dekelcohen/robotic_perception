from .bbox_providers import BBoxProvider

def get_bbox_provider(detector: str) -> BBoxProvider:
    detector = detector.lower()
    if detector == "gemini":
        from .gemini_bbox_provider import GeminiBBoxProvider
        return GeminiBBoxProvider()
    elif detector == "moondream":
        from .moondream_bbox_provider import MoondreamBBoxProvider
        return MoondreamBBoxProvider()
    else:
        raise ValueError(f"Unknown bbox detector: {detector}")
