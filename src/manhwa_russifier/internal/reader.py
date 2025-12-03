import cv2 as cv
import numpy as np
import easyocr
from .viewer import Page


class ManhwaReader:
    def __init__(self, gpu: bool = False, language: str = 'en'):
        self.reader = easyocr.Reader(
            lang_list=[language],
            gpu=gpu
        )

    def _extract_text_from_segment(self, image: np.ndarray, segment: np.ndarray) -> str:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = segment.reshape((-1, 1, 2)).astype(np.int32)
        cv.fillPoly(mask, [pts], (255, 255, 255))
        segmented_image = cv.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv.boundingRect(pts)
        if w == 0 or h == 0:
            return ""
        cropped_image = segmented_image[y:y+h, x:x+w]
        result = self.reader.readtext(
            cropped_image,
            detail=1
        )
        if not result:
            return ""
        texts = []
        for (_, text, confidence) in result:
            texts.append(text)
        return " ".join(texts).strip()

    def extract_texts(self, pages: list[Page]):
        for page in pages:
            for image_text in page.image_texts:
                for segment in image_text.segments:
                    extracted_text = self._extract_text_from_segment(
                        page.cvImage,
                        segment
                    )
                    image_text.extracted_text = extracted_text
