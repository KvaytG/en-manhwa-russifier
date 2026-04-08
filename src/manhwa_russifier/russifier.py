import time
import logging
import numpy as np
from .internal import ManhwaViewer
from .internal import ManhwaReader
from .internal import ManhwaTranslator
from .internal import ManhwaCleaner
from .internal import ManhwaTyper

logger = logging.getLogger("ManhwaRussifier")


class ManhwaRussifier:
    def __init__(self,
                 gpu: bool,
                 font_path: str):
        self.viewer = ManhwaViewer(gpu)
        self.reader = ManhwaReader(gpu)
        self.translator = ManhwaTranslator()
        self.cleaner = ManhwaCleaner()
        self.typer = ManhwaTyper(font_path)

    def russify(self, images: list[np.ndarray]) -> list[np.ndarray]:
        total_start = time.perf_counter()
        # 1. Детекция баблов
        start = time.perf_counter()
        pages = self.viewer.get_pages(images)
        logger.info(f"Detection finished: {time.perf_counter() - start:.3f}s")
        # 2. OCR (Чтение текста)
        start = time.perf_counter()
        self.reader.fill_pages_data(pages)
        for page in pages:
            for it in page.image_texts:
                if it.extracted_text:
                    it.extracted_text = it.extracted_text.lower()
        logger.info(f"OCR finished: {time.perf_counter() - start:.3f}s")
        # 3. Перевод
        start = time.perf_counter()
        self.translator.translate_pages(pages)
        logger.info(f"Translation finished: {time.perf_counter() - start:.3f}s")
        # 4. Удаление старого текста
        start = time.perf_counter()
        self.cleaner.clean_texts(pages)
        logger.info(f"Inpainting finished: {time.perf_counter() - start:.3f}s")
        # 5. Рендеринг нового текста
        start = time.perf_counter()
        self.typer.render_texts(pages)
        logger.info(f"Rendering finished: {time.perf_counter() - start:.3f}s")
        total_time = time.perf_counter() - total_start
        logger.info(f"TOTAL: {total_time:.3f}s for {len(images)} pages ({total_time / len(images):.3f}s/page)")
        return [page.cv_image for page in pages]
