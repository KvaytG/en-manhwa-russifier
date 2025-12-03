import numpy as np
import cv2
from .viewer import Page
from .simple_lama_inpainting import SimpleLama


class ManhwaCleaner:
    def __init__(self,
                 gpu: bool,
                 big_lama_model_path: str):
        self.gpu = gpu
        self.simpleLama = SimpleLama(big_lama_model_path, gpu)

    def clean_texts(self, pages: list[Page]) -> list[Page]:
        for page in pages:
            telea_mask = np.zeros(page.cvImage.shape[:2], dtype=np.uint8)
            lama_mask = np.zeros(page.cvImage.shape[:2], dtype=np.uint8)
            for image_text in page.image_texts:
                for segment in image_text.segments:
                    pts = segment.reshape((-1, 1, 2)).astype(np.int32)
                    if image_text.category == 'text_bubble':
                        cv2.fillPoly(telea_mask, [pts], 255)
                    else:
                        cv2.fillPoly(lama_mask, [pts], 255)
            if np.any(telea_mask):
                image = page.cvImage.copy()
                page.cvImage = cv2.inpaint(
                    src=image,
                    inpaintMask=telea_mask,
                    inpaintRadius=3,
                    flags=cv2.INPAINT_TELEA
                )
            if np.any(lama_mask):
                lama_input = cv2.cvtColor(page.cvImage, cv2.COLOR_BGR2RGB) if len(
                    page.cvImage.shape) == 3 else cv2.cvtColor(page.cvImage, cv2.COLOR_GRAY2RGB)
                result = self.simpleLama(lama_input, lama_mask)
                if len(page.cvImage.shape) == 2:
                    page.cvImage = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
                else:
                    page.cvImage = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return pages
