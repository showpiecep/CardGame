from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from transformers.models.sam import SamModel, SamProcessor
from transformers.pipelines import ZeroShotObjectDetectionPipeline

from cardgame.detector.utils.load_utils import get_boxes, load_image, refine_masks
from cardgame.detector.utils.result import DetectionResult


class Detector:
    def __init__(
        self,
        detector_id: Optional[str] = "IDEA-Research/grounding-dino-tiny",
        segmenter_id: Optional[str] = "facebook/sam-vit-base",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id

        self.object_detector = None
        self.segmentator = None
        self.seg_processor = None

    def _init_models(self):
        self.object_detector: ZeroShotObjectDetectionPipeline = pipeline(
            model=self.detector_id,
            task="zero-shot-object-detection",
            device=self.device,
        )
        self.segmentator: SamModel = AutoModelForMaskGeneration.from_pretrained(
            self.segmenter_id,
        ).to(self.device)

        self.seg_processor: SamProcessor = AutoProcessor.from_pretrained(
            self.segmenter_id,
        )

    def detect(
        self,
        image: Image.Image,
        labels: List[str],
        threshold: float = 0.3,
    ) -> DetectionResult:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """

        labels = [label if label.endswith(".") else label + "." for label in labels]

        results = self.object_detector(
            image, candidate_labels=labels, threshold=threshold
        )
        results = [DetectionResult.from_dict(result) for result in results]

        return results

    def segment(
        self,
        image: Image.Image,
        detection_results: DetectionResult,
        polygon_refinement: bool = False,
    ) -> List[DetectionResult]:
        """
        Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
        """
        boxes = get_boxes(detection_results)
        inputs = self.seg_processor(
            images=image,
            input_boxes=boxes,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.segmentator(**inputs)
        masks = self.seg_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )[0]

        masks = refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results

    def grounded_segmentation(
        self,
        image: Union[Image.Image, str],
        labels: List[str],
        threshold: float = 0.3,
        polygon_refinement: bool = False,
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        if isinstance(image, str):
            image = load_image(image)

        detections = self.detect(image, labels, threshold)
        detections = self.segment(image, detections, polygon_refinement)

        return np.array(image), detections
