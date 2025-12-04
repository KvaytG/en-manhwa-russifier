# manhwa-russifier

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![MIT License](https://img.shields.io/badge/License-MIT-green)

A complete pipeline for automatic russification of manhwa: text detection, OCR, translation, replacement, and rendering.

## 📚 Usage

```python
import os
import asyncio
import cv2
import torch
from manhwa_russifier import ManhwaRussifier, pack_cv_images, PackMethod


async def main():
    input_folder = "input"
    os.makedirs(input_folder, exist_ok=True)
    files = sorted(os.listdir(input_folder))
    images = []
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            path = os.path.join(input_folder, f)
            img = cv2.imread(path)
            if img is None:
                print(f"Failed to read {f}")
                continue
            images.append(img)
    if not images:
        print("No images found!")
        return
    russifier = ManhwaRussifier(
        gpu=torch.cuda.is_available(),
        # Path to bubble detector
        # Download: https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m
        bubble_detector_path='assets/detection/comic-speech-bubble-detector.pt',
        # Path to text segmenter
        # Download: https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m
        text_detector_path='assets/detection/comic-text-segmenter.pt',
        # Path to LaMa
        # Download: https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
        big_lama_model_path='assets/inpainting/big-lama.pt',
        # Path to the font to use
        font_path='assets/fonts/your-font.ttf',
        source_language='en'
    )
    # Processing
    rendered_images = await russifier.russify(images)
    # Saving. Supports PDF, ZIP, 7ZIP, CBZ, CB7 and other formats specified in PackMethod
    pack_cv_images(rendered_images, "output", PackMethod.PDF)

if __name__ == "__main__":
    asyncio.run(main())

```

## ⚙️ Installation
```bash
pip install git+https://github.com/KvaytG/manhwa-russifier.git
```

## ⚠️ Notice
This project is **experimental** and is no longer maintained.  
No new features or bug fixes will be implemented.  
Use at your own risk.

## 📜 License
Licensed under the **[MIT](LICENSE.txt)** license.

This project uses open-source components. For license details see **[pyproject.toml](pyproject.toml)** and dependencies' official websites.
