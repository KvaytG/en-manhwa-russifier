import cv2
import numpy as np
from PIL import Image
from caption_forge import generate_caption_image
from .viewer import Page


class ManhwaTyper:
    def __init__(self, font_path: str):
        self.font_path = font_path

    @staticmethod
    def _calculate_bounding_box_from_segments(segments: list[np.ndarray]) -> tuple[int, int, int, int]:
        if not segments:
            return 0, 0, 0, 0
        all_points = np.vstack(segments)
        x_min = int(np.min(all_points[:, 0]))
        y_min = int(np.min(all_points[:, 1]))
        x_max = int(np.max(all_points[:, 0]))
        y_max = int(np.max(all_points[:, 1]))
        padding_x = max(10, int((x_max - x_min) * 0.1))
        padding_y = max(10, int((y_max - y_min) * 0.1))
        return (
            max(0, x_min - padding_x),
            max(0, y_min - padding_y),
            x_max + padding_x,
            y_max + padding_y
        )

    def render_texts(self, pages: list[Page]) -> list[Page]:
        for page in pages:
            for image_text in page.image_texts:
                if not image_text.extracted_text or not image_text.extracted_text.strip():
                    continue

                x1, y1, x2, y2 = self._calculate_bounding_box_from_segments(image_text.segments)
                bubble_width = x2 - x1
                bubble_height = y2 - y1

                if bubble_width < 30 or bubble_height < 20:
                    continue

                temp_image = Image.new('RGB', (bubble_width, bubble_height), color='white')

                try:
                    rendered_text_image = generate_caption_image(
                        pil_image=temp_image,
                        text=image_text.extracted_text,
                        font_path=self.font_path
                    )

                    rendered_np = np.array(rendered_text_image)
                    if len(rendered_np.shape) == 3:
                        rendered_np = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)

                    text_mask = cv2.cvtColor(rendered_np, cv2.COLOR_BGR2GRAY)
                    _, text_mask = cv2.threshold(text_mask, 200, 255, cv2.THRESH_BINARY)
                    text_area_mask = cv2.bitwise_not(text_mask)

                    roi = page.cvImage[y1:y2, x1:x2]

                    if roi.shape[:2] == rendered_np.shape[:2]:
                        background = cv2.bitwise_and(roi, roi, mask=text_mask)
                        foreground = cv2.bitwise_and(rendered_np, rendered_np, mask=text_area_mask)
                        page.cvImage[y1:y2, x1:x2] = cv2.add(background, foreground)
                    else:
                        resized_rendered = cv2.resize(rendered_np, (roi.shape[1], roi.shape[0]))
                        resized_text_mask = cv2.resize(text_mask, (roi.shape[1], roi.shape[0]))
                        resized_text_area_mask = cv2.resize(text_area_mask, (roi.shape[1], roi.shape[0]))
                        background = cv2.bitwise_and(roi, roi, mask=resized_text_mask)
                        foreground = cv2.bitwise_and(resized_rendered, resized_rendered, mask=resized_text_area_mask)
                        page.cvImage[y1:y2, x1:x2] = cv2.add(background, foreground)

                except Exception as e:
                    print(f"Ошибка при рендеринге текста: {e}")
                    continue

        return pages