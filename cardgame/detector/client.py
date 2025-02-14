from dataclasses import dataclass
import datetime
from typing import Optional

import numpy as np
import requests
from PIL import Image

from cardgame.detector.utils.load_utils import decode_base64_image, image_to_base64
from cardgame.detector.utils.result import DetectionResult


@dataclass
class DetectorClientResponse:
    annotated_data: Image.Image
    detections: list[DetectionResult]


class DetectorClient:
    def __init__(self):
        self._url = "http://127.0.0.1:8000/segment"

    def predict(
        self,
        image: Image.Image,
        labels: Optional[list[str]] = None,
    ) -> DetectorClientResponse:
        encoded_image = image_to_base64((np.array(image)))
        payload = {
            "image_base64": encoded_image,
            "labels": labels or ["cards"],
            "threshold": 0.3,
            "polygon_refinement": False,
        }
        response = requests.post(self._url, json=payload)
        if response.status_code == 200:
            data = response.json()

            annotated_image_b64 = data.get("image_base64")
            detections = data.get("detected")

            decoded_annotated_image = decode_base64_image(annotated_image_b64)
        else:
            raise RuntimeError

        return DetectorClientResponse(
            decoded_annotated_image,
            [DetectionResult.from_dict(item) for item in detections],
        )


if __name__ == "__main__":
    image = Image.open(
        requests.get("https://www.pagat.com/images/beating/durak.jpg", stream=True).raw
    ).convert("RGB")
    start = datetime.now()
    tmp = DetectorClient().predict(image)
    elapsed = datetime.now() - start

    seconds = elapsed.total_seconds()
    print(f"Execution time: {seconds:.4f} seconds")