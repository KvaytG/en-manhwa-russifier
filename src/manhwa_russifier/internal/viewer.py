import cv2 as cv
import numpy as np
from ultralytics import YOLO


def calculate_iou(rect1, rect2) -> float:
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    rect1area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    rect2area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union_area = rect1area + rect2area - intersection_area
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou


def do_rectangles_overlap(rect1, rect2, iou_threshold: float = 0.2) -> bool:
    iou = calculate_iou(rect1, rect2)
    overlap = iou >= iou_threshold
    return overlap


def does_rectangle_fit(bigger_rect, smaller_rect):
    x1, y1, x2, y2 = bigger_rect
    px1, py1, px2, py2 = smaller_rect
    left1, top1, right1, bottom1 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    left2, top2, right2, bottom2 = min(px1, px2), min(py1, py2), max(px1, px2), max(py1, py2)
    fits_horizontally = left1 <= left2 and right1 >= right2
    fits_vertically = top1 <= top2 and bottom1 >= bottom2
    return fits_horizontally and fits_vertically


def combine_results(bubble_detect_results, text_seg_results):
    bubble_bounding_boxes = np.array(bubble_detect_results.boxes.xyxy.cpu(), dtype="int")
    text_bounding_boxes = np.array(text_seg_results.boxes.xyxy.cpu(), dtype="int")
    segment_points = []
    if text_seg_results.masks is not None:
        segment_points = [mask.astype("int") for mask in text_seg_results.masks.xy]
    raw_results = []
    text_matched = [False] * len(text_bounding_boxes)
    if segment_points:
        for txtIdx, txtBox in enumerate(text_bounding_boxes):
            for bbleBox in bubble_bounding_boxes:
                if does_rectangle_fit(bbleBox, txtBox):
                    raw_results.append((txtBox, bbleBox, segment_points[txtIdx], 'text_bubble'))
                    text_matched[txtIdx] = True
                    break
                elif do_rectangles_overlap(bbleBox, txtBox):
                    raw_results.append((txtBox, bbleBox, segment_points[txtIdx], 'text_free'))
                    text_matched[txtIdx] = True
                    break
    return raw_results


def _ensure_rgb_image(cv_image: np.ndarray) -> np.ndarray:
    if len(cv_image.shape) == 2:
        cv_image = cv.cvtColor(cv_image, cv.COLOR_GRAY2RGB)
    elif cv_image.shape[2] == 4:
        cv_image = cv.cvtColor(cv_image, cv.COLOR_RGBA2RGB)
    elif cv_image.shape[2] == 3:
        cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
    return cv_image


def _resize_cv_image_to_32x(cv_image: np.ndarray) -> np.ndarray:
    height, width = cv_image.shape[:2]
    if width % 32 == 0 and height % 32 == 0:
        return cv_image
    new_width = max(32, (width // 32) * 32)
    new_height = max(32, (height // 32) * 32)
    return cv.resize(cv_image, (new_width, new_height))


class ImageText:
    def __init__(self,
                 segments: list[np.ndarray],
                 bubble_box: tuple[int, int, int, int],
                 category: str):
        self.segments = segments
        self.bubbleBox = bubble_box
        self.category = category
        self.extracted_text = None

    def get_height(self) -> int:
        return self.bubbleBox[3] - self.bubbleBox[1]


class Page:
    def __init__(self, cv_image: np.ndarray, image_texts: list['ImageText']):
        self.cvImage = cv_image
        self.image_texts = image_texts
        self.cvMask = self._create_mask()

    def _create_mask(self) -> np.ndarray:
        mask = np.zeros(self.cvImage.shape[:2], dtype=np.uint8)
        for image_text in self.image_texts:
            for segment in image_text.segments:
                pts = segment.reshape((-1, 1, 2)).astype(np.int32)
                cv.fillPoly(mask, [pts], 255)
        return mask


class ManhwaViewer:
    def __init__(self,
                 gpu: bool,
                 bubble_detector_path: str,
                 text_detector_path: str):
        self.gpu = gpu
        self.yoloDevice = '0' if gpu else 'cpu'
        self.bubbleDetector = YOLO(bubble_detector_path)
        self.textSegmenter = YOLO(text_detector_path)

    def get_pages(self, cv_images: list[np.ndarray]) -> list[Page]:
        pages = []
        for cv_image in cv_images:
            orig_image = cv_image.copy()
            detection_image = _ensure_rgb_image(cv_image)
            detection_image = _resize_cv_image_to_32x(detection_image)
            height, width = detection_image.shape[:2]
            size = max(height, width)
            bubble_results = self.bubbleDetector(detection_image, device=self.yoloDevice, imgsz=size, conf=0.1,
                                                 verbose=False)
            text_results = self.textSegmenter(detection_image, device=self.yoloDevice, imgsz=size, conf=0.1,
                                              verbose=False)
            combined_results = combine_results(bubble_results[0], text_results[0])
            image_texts = []
            for textBox, bubbleBox, segments, category in combined_results:
                image_text = ImageText(
                    [segments],
                    bubbleBox,
                    category
                )
                image_texts.append(image_text)
            image_texts.sort(key=lambda x: x.bubbleBox[1])
            pages.append(Page(orig_image, image_texts))

        return pages
