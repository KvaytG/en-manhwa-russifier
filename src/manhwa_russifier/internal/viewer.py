import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download


class ImageText:
    def __init__(self, bbox, segments=None, extracted_text=None):
        self.bbox = bbox  # (x, y, w, h)
        self.segments = segments if segments is not None else []
        self.extracted_text = extracted_text


class Page:
    def __init__(self, cv_image, image_texts):
        self.cv_image = cv_image
        self.image_texts = image_texts
        self.cv_mask = None

    def create_mask(self):
        mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
        for it in self.image_texts:
            for seg in it.segments:
                pts = seg.reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [pts], (255, 255, 255))
        self.cv_mask = mask


class ManhwaViewer:
    def __init__(self, gpu: bool, conf_threshold: float = 0.3):
        detector_path = hf_hub_download(
            repo_id="ogkalu/comic-text-and-bubble-detector",
            filename="detector.onnx"
        )
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(detector_path, providers=providers)
        self.conf_threshold = conf_threshold
        self.input_size = (640, 640)

    def get_pages(self, cv_images: list[np.ndarray]) -> list[Page]:
        pages = []
        for img in cv_images:
            if img is None: continue
            bboxes, scores = self._detect_text(img)
            image_texts = []
            for bbox in bboxes:
                it = ImageText(bbox=tuple(bbox))
                image_texts.append(it)
            image_texts.sort(key=lambda x: x.bbox[1])
            pages.append(Page(img.copy(), image_texts))
        return pages

    def _detect_text(self, img):
        h_orig, w_orig = img.shape[:2]
        img_resized = cv2.resize(img, self.input_size)
        blob = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        input_feed = {
            "images": blob,
            "orig_target_sizes": np.array([[w_orig, h_orig]], dtype=np.int64)
        }
        outputs = self.session.run(None, input_feed)
        labels, boxes, scores = outputs[0][0], outputs[1][0], outputs[2][0]
        raw_boxes, raw_scores = [], []
        for i in range(len(scores)):
            if scores[i] >= self.conf_threshold and labels[i] == 1:
                x1, y1, x2, y2 = boxes[i]
                raw_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                raw_scores.append(float(scores[i]))
        if not raw_boxes: return [], []
        nms_indices = cv2.dnn.NMSBoxes(raw_boxes, raw_scores, self.conf_threshold, 0.45)
        if len(nms_indices) == 0: return [], []
        nms_boxes = [raw_boxes[i] for i in nms_indices.flatten()]
        final_indices = self._filter_nested_boxes(nms_boxes)
        return [nms_boxes[i] for i in final_indices], [raw_scores[i] for i in final_indices]

    @staticmethod
    def _filter_nested_boxes(boxes, threshold=0.8):
        if not boxes: return []
        keep = np.ones(len(boxes), dtype=bool)
        b = np.array(boxes)  # [x, y, w, h]
        areas = b[:, 2] * b[:, 3]
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j or not keep[i] or not keep[j]: continue
                xx1 = max(b[i, 0], b[j, 0])
                yy1 = max(b[i, 1], b[j, 1])
                xx2 = min(b[i, 0] + b[i, 2], b[j, 0] + b[j, 2])
                yy2 = min(b[i, 1] + b[i, 3], b[j, 1] + b[j, 3])
                w, h = max(0, xx2 - xx1), max(0, yy2 - yy1)
                inter_area = w * h
                if inter_area > 0:
                    ioa_i = inter_area / areas[i]
                    ioa_j = inter_area / areas[j]

                    if ioa_i > threshold:
                        keep[i] = False
                        break
                    elif ioa_j > threshold:
                        keep[j] = False
        return [i for i, val in enumerate(keep) if val]
