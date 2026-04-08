import cv2 as cv
import numpy as np
from simple_image_inpainter import SimpleImageInpainter
from .viewer import Page


class ManhwaCleaner:
    def __init__(self):
        self.inpainter = SimpleImageInpainter(dilation_size=3)

    def _clean_page(self, page: Page):
        page.create_mask()
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.dilate(page.cv_mask, kernel, iterations=2)
        page.cv_image = self.inpainter.inpaint(page.cv_image, mask)

    def clean_texts(self, pages: list[Page]):
        for page in pages:
            self._clean_page(page)
