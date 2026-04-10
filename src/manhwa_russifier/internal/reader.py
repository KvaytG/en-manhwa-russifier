import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import TrOCRProcessor
from .viewer import Page

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import transformers

transformers.logging.set_verbosity_error()


class ManhwaReader:
    def __init__(self,
                 gpu: bool,
                 batch_size: int = 16):
        trocr_repo = "KvaytG/trocr-base-printed-onnx"
        craft_repo = "KvaytG/craft-mlt-25k-onnx"
        self.batch_size = batch_size
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
        self.device = "cuda" if gpu else "cpu"
        craft_path = hf_hub_download(repo_id=craft_repo, filename="craft.onnx")
        self.craft_session = ort.InferenceSession(craft_path, providers=self.providers)
        self.craft_input_name = self.craft_session.get_inputs()[0].name
        self.processor = TrOCRProcessor.from_pretrained(trocr_repo)
        self.trocr_model = ORTModelForVision2Seq.from_pretrained(
            trocr_repo,
            providers=self.providers,
            use_cache=True,
            decoder_file_name="decoder_model.onnx",
            decoder_with_past_file_name="decoder_with_past_model.onnx"
        )

    def _get_craft_segments(self, roi, offset):
        x_off, y_off = offset
        h_roi, w_roi = roi.shape[:2]
        target_h, target_w = int(np.ceil(h_roi / 32) * 32), int(np.ceil(w_roi / 32) * 32)
        img_resized = cv2.resize(roi, (target_w, target_h))
        img_in = img_resized.transpose(2, 0, 1).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0) / 255.0
        y_out = self.craft_session.run(None, {self.craft_input_name: img_in})[0]
        score_text = cv2.resize(y_out[0, :, :, 0], (w_roi, h_roi))
        _, thresh = cv2.threshold(score_text, 0.25, 1.0, cv2.THRESH_BINARY)
        thresh = (thresh * 255).astype(np.uint8)
        thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)), iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segments = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 5: continue
            cnt_flat = cnt.reshape(-1, 2).astype(np.float32)
            cnt_flat[:, 0] += x_off
            cnt_flat[:, 1] += y_off
            segments.append(cnt_flat.astype(np.int32))
        return segments

    def _batch_ocr(self, image_crops: list[np.ndarray]) -> list[str]:
        if not image_crops: return []
        results = []
        for i in range(0, len(image_crops), self.batch_size):
            batch = image_crops[i: i + self.batch_size]
            pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in batch]
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.trocr_model.generate(
                    inputs.pixel_values,
                    max_new_tokens=64,
                    use_cache=True
                )
            texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            results.extend([t.strip() for t in texts])
        return results

    def fill_pages_data(self, pages: list[Page]):
        for page in pages:
            img = page.cv_image
            all_line_crops = []
            it_line_map = []
            for it in page.image_texts:
                x, y, w, h = it.bbox
                pad = 2
                y1, y2 = max(0, y - pad), min(img.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(img.shape[1], x + w + pad)
                roi = img[y1:y2, x1:x2]
                if roi.size == 0: continue
                it.segments = self._get_craft_segments(roi, offset=(x1, y1))
                if not it.segments:
                    it.extracted_text = ""
                    continue
                boxes = [cv2.boundingRect(seg) for seg in it.segments]
                boxes.sort(key=lambda b: b[1])
                lines = []
                current_line = [boxes[0]]
                for box in boxes[1:]:
                    last_box = current_line[-1]
                    if abs(box[1] - last_box[1]) < max(box[3], last_box[3]) * 0.5:
                        current_line.append(box)
                    else:
                        lines.append(current_line)
                        current_line = [box]
                if current_line: lines.append(current_line)
                for line in lines:
                    lx_min = min(b[0] for b in line)
                    ly_min = min(b[1] for b in line)
                    lx_max = max(b[0] + b[2] for b in line)
                    ly_max = max(b[1] + b[3] for b in line)
                    line_pad = 5
                    lx_min = max(0, lx_min - line_pad)
                    ly_min = max(0, ly_min - line_pad)
                    lx_max = min(img.shape[1], lx_max + line_pad)
                    ly_max = min(img.shape[0], ly_max + line_pad)
                    line_roi = img[ly_min:ly_max, lx_min:lx_max]
                    if line_roi.size > 0:
                        pad_w = 5
                        line_roi_padded = cv2.copyMakeBorder(
                            line_roi, pad_w, pad_w, pad_w, pad_w,
                            cv2.BORDER_CONSTANT, value=[255, 255, 255]
                        )
                        all_line_crops.append(line_roi_padded)
                        it_line_map.append(it)
            if all_line_crops:
                translated_lines = self._batch_ocr(all_line_crops)
                results_per_it = {}
                for it_obj, text in zip(it_line_map, translated_lines):
                    if it_obj not in results_per_it:
                        results_per_it[it_obj] = []
                    if text:
                        results_per_it[it_obj].append(text)
                for it_obj, lines_list in results_per_it.items():
                    it_obj.extracted_text = " ".join(lines_list)
