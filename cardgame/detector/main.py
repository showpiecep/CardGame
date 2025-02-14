from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, HTTPException

from cardgame.detector.api.models import DetectorResponse, SegmentationRequest
from cardgame.detector.core.detector import Detector
from cardgame.detector.utils.load_utils import decode_base64_image, image_to_base64
from cardgame.detector.utils.plot import annotate
from cardgame.detector.utils.serialize import convert_numpy

app = FastAPI(title="Detector API")

detector_instance = Detector()


# Инициализация моделей при старте приложения
@app.on_event("startup")
def load_models():
    detector_instance._init_models()


@app.post("/segment")
def segment_endpoint(request: SegmentationRequest) -> DetectorResponse:
    # Декодирование base64 изображения
    image = decode_base64_image(request.image_base64)

    try:
        image_array, detections = detector_instance.grounded_segmentation(
            image=image,
            labels=request.labels,
            threshold=request.threshold,
            polygon_refinement=request.polygon_refinement,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {e}")

    # Конвертируем итоговое изображение в base64 для передачи в JSON
    annotated_image = annotate(image_array, detections)
    annotated_image_b64 = image_to_base64(annotated_image)

    return DetectorResponse(
        image_base64=annotated_image_b64,
        detected=[convert_numpy(asdict(detection)) for detection in detections],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
