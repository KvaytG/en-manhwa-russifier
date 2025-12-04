# manhwa-russifier

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![MIT License](https://img.shields.io/badge/License-MIT-green)

Полный пайплайн для автоматической русификации манхвы: детекция текста, OCR, перевод, замена и рендеринг.

## 📚 Использование

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
                print(f"Не удалось прочитать {f}")
                continue
            images.append(img)
    if not images:
        print("Изображения не найдены!")
        return
    russifier = ManhwaRussifier(
        gpu=torch.cuda.is_available(),
        # Путь к детектору пузырей
        # Скачать: https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m
        bubble_detector_path='assets/detection/comic-speech-bubble-detector.pt',
        # Путь к сегментатору текста
        # Скачать: https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m
        text_detector_path='assets/detection/comic-text-segmenter.pt',
        # Путь к LaMa
        # Скачать: https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
        big_lama_model_path='assets/inpainting/big-lama.pt',
        # Путь к вашему шрифту
        font_path='assets/fonts/your-font.ttf',
        source_language='en'
    )
    # Обработка
    rendered_images = await russifier.russify(images)
    # Сохранение. Поддерживает форматы PDF, ZIP, 7ZIP, CBZ, CB7 и другие, указанные в PackMethod
    pack_cv_images(rendered_images, "output", PackMethod.PDF)

if __name__ == "__main__":
    asyncio.run(main())

```

## ⚙️ Установка

```bash
pip install git+https://github.com/KvaytG/manhwa-russifier.git
```

## ⚠️ Важно

Этот проект **экспериментальный** и больше не поддерживается.
Новые функции и исправления ошибок добавляться не будут.
Используйте на свой страх и риск.

## 📜 Лицензия

Распространяется по лицензии **[MIT](LICENSE.txt)**.

Проект использует компоненты с открытым исходным кодом. Сведения о лицензиях см. в **[pyproject.toml](pyproject.toml)** и на официальных ресурсах зависимостей.
