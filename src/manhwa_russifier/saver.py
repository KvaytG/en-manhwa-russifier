import os
import random
import shutil
import string
import tarfile
import tempfile
from enum import Enum
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image
from py7zr import SevenZipFile
import cv2 as cv
import numpy as np


class PackMethod(Enum):
    CBZ = '.cbz'
    CB7 = '.cb7'
    CBT = '.cbt'
    ZIP = '.zip'
    SEVEN_ZIP = '.7z'
    TAR = '.tar'
    PDF = '.pdf'
    NONE = ''


def _get_random_chars(length: int) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def pack_cv_images(cv_images: list[np.ndarray], output_path: str, pack_method: PackMethod):
    temp_dir = tempfile.mkdtemp()
    num_digits = len(str(len(cv_images)))
    image_paths = []

    for i, cv_image in enumerate(cv_images, start=1):
        image_path = os.path.join(temp_dir, f"{i:0{num_digits}}.png")
        cv.imwrite(image_path, cv_image)
        image_paths.append(image_path)

    output_path += pack_method.value

    while os.path.exists(output_path):
        base_name, extension = os.path.splitext(output_path)
        output_path = f'{base_name}_{_get_random_chars(8)}{extension}'

    if pack_method in (PackMethod.CBZ, PackMethod.ZIP):
        with ZipFile(output_path, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for image_path in image_paths:
                archive.write(image_path, os.path.basename(image_path))

    elif pack_method in (PackMethod.CB7, PackMethod.SEVEN_ZIP):
        with SevenZipFile(output_path, 'w') as archive:
            for image_path in image_paths:
                archive.write(image_path, os.path.basename(image_path))

    elif pack_method in (PackMethod.CBT, PackMethod.TAR):
        with tarfile.open(output_path, 'w') as archive:
            for image_path in image_paths:
                archive.add(image_path, os.path.basename(image_path))

    elif pack_method == PackMethod.PDF:
        pil_images = [Image.open(image_path) for image_path in image_paths]
        pil_images[0].save(
            output_path,
            save_all=True,
            append_images=pil_images[1:],
            resolution=100,
            optimize=True
        )

    elif pack_method == PackMethod.NONE:
        shutil.move(temp_dir, output_path)
        return

    shutil.rmtree(temp_dir)
