# en-manhwa-russifier

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![MIT License](https://img.shields.io/badge/License-MIT-green)

A complete pipeline for automatic russification of English manhwa: text detection, OCR, translation, replacement, and rendering.

## 📚 Usage

```python
import os
import cv2
import torch
from manhwa_russifier import pack_cv_images, PackMethod, ManhwaRussifier


def main(input_path: str, output_path: str):
    os.makedirs(input_path, exist_ok=True)
    files = sorted(os.listdir(input_path))
    images = []
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg",)):
            path = os.path.join(input_path, f)
            img = cv2.imread(path)
            if img is None:
                print(f"Failed to read {f}")
                continue
            images.append(img)
    if not images:
        print("Images not found")
        return
    russifier = ManhwaRussifier(
        gpu=torch.cuda.is_available(),
        font_path='path/to/your/font.ttf'
    )
    rendered_images = russifier.russify(images)
    # Saving. Supports PDF, ZIP, 7ZIP, CBZ, CB7 and other formats specified in PackMethod
    pack_cv_images(rendered_images, output_path, PackMethod.CBZ)


if __name__ == "__main__":
    main('input', 'output')
    
```

## ⚙️ Installation
```bash
pip install git+https://github.com/KvaytG/en-manhwa-russifier.git
```

## ⚠️ Notice
This project is **experimental**.  
Use at your own risk.

## 📜 License
Licensed under the **[MIT](LICENSE.txt)** license.

This project uses open-source components. For license details see **[pyproject.toml](pyproject.toml)** and dependencies' official websites.
