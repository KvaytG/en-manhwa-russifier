import re
from en_ru_translator import Translator
from .viewer import Page

_HYPHEN_PATTERN = re.compile(r'(\w+)-\s*\n?\s*(\w+)')
_SPACE_PATTERN = re.compile(r'\s+')
_DOT_PATTERN = re.compile(r'\.{2,}')


def _prepare_src_text(text: str) -> str:
    text = _HYPHEN_PATTERN.sub(r'\1\2', text)
    text = _SPACE_PATTERN.sub(' ', text)
    return text.strip()


def _prepare_tgt_text(text: str) -> str:
    text = _DOT_PATTERN.sub('...', text)
    text = _SPACE_PATTERN.sub(' ', text)
    return text.strip()


class ManhwaTranslator:
    def __init__(self, batch_size: int = 16):
        self.translator = Translator()
        self.batch_size = batch_size

    def translate_pages(self, pages: list[Page]):
        all_elements = []
        texts_to_translate = []
        for page in pages:
            for img_text in page.image_texts:
                src = _prepare_src_text(img_text.extracted_text)
                if not src:
                    continue
                all_elements.append(img_text)
                texts_to_translate.append(src)
        if texts_to_translate:
            all_translated = []
            for i in range(0, len(texts_to_translate), self.batch_size):
                batch = texts_to_translate[i: i + self.batch_size]
                translated_batch = self.translator.translate_batch(batch)
                all_translated.extend(translated_batch)
            for img_text, tr_text in zip(all_elements, all_translated):
                img_text.extracted_text = _prepare_tgt_text(tr_text)
