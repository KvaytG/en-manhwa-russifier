import cv2
import numpy as np
from PIL import Image
from caption_forge import generate_caption_image
from .viewer import Page


class ManhwaTyper:
    def __init__(self, font_path: str):
        self.font_path = font_path

    def render_texts(self, pages: list[Page]):
        for page in pages:
            for image_text in page.image_texts:
                text = image_text.extracted_text
                if not text or not text.strip() or not image_text.segments:
                    continue
                x1, y1, x2, y2 = self._calculate_bounding_box_from_segments(image_text.segments)
                bw, bh = x2 - x1, y2 - y1
                if bw < 10 or bh < 10: continue
                temp_canvas = Image.new('RGBA', (bw, bh), (255, 255, 255, 0))
                rendered_pil = generate_caption_image(
                    pil_image=temp_canvas,
                    text=text,
                    font_path=self.font_path
                ).convert("RGBA")
                page_pil = Image.fromarray(cv2.cvtColor(page.cv_image, cv2.COLOR_BGR2RGB)).convert("RGBA")
                page_pil.paste(rendered_pil, (x1, y1), rendered_pil)
                page.cv_image[:] = cv2.cvtColor(np.array(page_pil), cv2.COLOR_RGBA2BGR)

    @staticmethod
    def _calculate_bounding_box_from_segments(segments):
        if not segments: return 0, 0, 0, 0
        all_pts = np.concatenate(segments)
        x_min, y_min = np.min(all_pts, axis=0)
        x_max, y_max = np.max(all_pts, axis=0)
        return int(x_min), int(y_min), int(x_max), int(y_max)
