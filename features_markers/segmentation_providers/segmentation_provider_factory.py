from .segmentation_providers import ISegmentation


def get_segmentation_provider(name: str) -> ISegmentation:
    name = (name or "").lower()
    if name in ("sam3", "roboflow-sam3", "sam"):
        from .sam3_segmentation_provider import SAM3SegmentationProvider
        return SAM3SegmentationProvider()
    if name in ("moondream", "md", "moondreamvl"):
        from features_markers.bbox_providers.moondream_bbox_provider import MoondreamVLMProvider
        return MoondreamVLMProvider()
    raise ValueError(f"Unknown segmentation provider: {name}")
