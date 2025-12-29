from .segmentation_providers import ISegmentation


def get_segmentation_provider(name: str) -> ISegmentation:
    name = (name or "").lower()
    if name in ("sam3", "roboflow-sam3", "sam"):
        from .sam3_segmentation_provider import SAM3SegmentationProvider
        return SAM3SegmentationProvider()
    raise ValueError(f"Unknown segmentation provider: {name}")

