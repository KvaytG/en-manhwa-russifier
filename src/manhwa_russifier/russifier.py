import numpy as np
from .internal import ManhwaTranslator
from .internal import ManhwaCleaner
from .internal import ManhwaReader
from .internal import ManhwaTyper
from .internal import ManhwaViewer

class ManhwaRussifier:
    def __init__(self,
                 gpu: bool,
                 bubble_detector_path: str,
                 text_detector_path: str,
                 big_lama_model_path: str,
                 font_path: str,
                 source_language: str = 'en'):
        self.source_language = source_language
        self.viewer = ManhwaViewer(
            gpu=gpu,
            bubble_detector_path=bubble_detector_path,
            text_detector_path=text_detector_path
        )
        self.cleaner = ManhwaCleaner(
            gpu=gpu,
            big_lama_model_path=big_lama_model_path
        )
        self.reader = ManhwaReader(
            gpu=gpu,
            language=source_language
        )
        self.typer = ManhwaTyper(font_path)
        self.translator = ManhwaTranslator()

    async def russify(self, images: list[np.ndarray]) -> list[np.ndarray]:
        pages = self.viewer.get_pages(images)
        self.reader.extract_texts(pages)
        tasks = self.translator.schedule_translate_texts(pages, self.source_language)
        cleaned_pages = self.cleaner.clean_texts(pages)
        await self.translator.finalize_translations(tasks)
        rendered_pages = self.typer.render_texts(cleaned_pages)
        return [page.cvImage for page in rendered_pages]