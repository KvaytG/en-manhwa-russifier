import torch
import numpy as np
import cv2 as cv
from .util import prepare_img_and_mask


class SimpleLama:
    def __init__(self, model_path: str, gpu: bool) -> None:
        device = torch.device('cuda' if gpu else 'cpu')
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        original_size = (image.shape[1], image.shape[0])
        image, mask = prepare_img_and_mask(image, mask, self.device)
        with torch.inference_mode():
            inpainted = self.model(image, mask)
            currentResult = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            currentResult = np.clip(currentResult * 255, 0, 255).astype(np.uint8)
            currentResult = cv.convertScaleAbs(currentResult)
            currentResult = cv.resize(currentResult, original_size, interpolation=cv.INTER_CUBIC)
            return currentResult
