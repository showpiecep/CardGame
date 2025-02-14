from typing import List, Optional

from pydantic import BaseModel


class SegmentationRequest(BaseModel):
    image_base64: str  # base64-кодированное изображение
    labels: List[str]
    threshold: Optional[float] = 0.3
    polygon_refinement: Optional[bool] = False


class SerializableDetectionResult(BaseModel):
    score: float
    label: str
    box: dict[str, int]
    mask: list[float] = None


class DetectorResponse(BaseModel):
    image_base64: str
    detected: list[dict]
    # detected: list[SerializableDetectionResult]
    # tmp: list[DetectionResult]
