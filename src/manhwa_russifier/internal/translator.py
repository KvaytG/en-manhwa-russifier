import asyncio
import json
import os
from collections import OrderedDict
from googletrans import Translator
from .viewer import Page

CACHE_FILE = "cache.json"
CACHE_SIZE = 1000


class LRUCache:
    def __init__(self, size: int, path: str):
        self.size = size
        self.path = path
        self.data = OrderedDict()
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.data = OrderedDict(raw)
            except Exception:
                self.data = OrderedDict()

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get(self, key):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        return None

    def set(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.size:
            self.data.popitem(last=False)
        self._save()


class ManhwaTranslator:
    def __init__(self):
        self.translator = Translator()
        self.cache = LRUCache(CACHE_SIZE, CACHE_FILE)

    async def translate(self, text: str, language: str) -> str:
        key = f"{language}:{text}"

        cached = self.cache.get(key)
        if cached is not None:
            return cached

        result = await self.translator.translate(text, src=language, dest="ru")
        translated = result.text

        self.cache.set(key, translated)
        return translated

    def schedule_translate_texts(self, pages: list[Page], language: str):
        tasks = []
        for page in pages:
            for image_text in page.image_texts:
                src = image_text.extracted_text
                if not src:
                    continue
                task = asyncio.create_task(self.translate(src, language))
                tasks.append((task, image_text))
        return tasks

    @staticmethod
    async def finalize_translations(tasks):
        for task, image_text in tasks:
            translated = await task
            image_text.extracted_text = translated
