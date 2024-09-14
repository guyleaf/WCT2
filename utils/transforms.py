from typing import Union

import numpy as np
from PIL import Image, ImageOps


def pad_to_divisible(
    image: Union[np.ndarray, Image.Image], divisor: int = 8, color=(0, 0, 0)
):
    pil_image = image
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)

    # calculate border length
    right = (divisor - pil_image.size[0] % divisor) % divisor
    bottom = (divisor - pil_image.size[1] % divisor) % divisor

    if right != 0 or bottom != 0:
        padded_image = ImageOps.expand(pil_image, (0, 0, right, bottom), fill=color)
    else:
        padded_image = pil_image.copy()

    if isinstance(image, np.ndarray):
        return np.array(padded_image)
    return padded_image


def remove_pad(
    image: Union[np.ndarray, Image.Image], target_width: int, target_height: int
):
    pil_image = image
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)

    # calculate border length
    right = pil_image.size[0] - target_width
    bottom = pil_image.size[1] - target_height
    assert right >= 0 and bottom >= 0

    if right != 0 or bottom != 0:
        cropped_image = ImageOps.crop(pil_image, (0, 0, right, bottom))
    else:
        cropped_image = pil_image.copy()

    if isinstance(image, np.ndarray):
        return np.array(cropped_image)
    return cropped_image
