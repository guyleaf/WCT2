"""
from photo_wct.py of https://github.com/NVIDIA/FastPhotoStyle
Copyright (C) 2018 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0
"""

import datetime
import mimetypes
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

from .transforms import pad_to_divisible, remove_pad


class Timer:
    def __init__(self, msg="Elapsed time: {}", verbose=True):
        self.msg = msg
        self.start_time = None
        self.verbose = verbose

    def __enter__(self):
        self.start_time = datetime.datetime.now()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.verbose:
            print(self.msg.format(datetime.datetime.now() - self.start_time))


def open_image(image_path: Union[str, Path], image_size: Optional[int] = None):
    image = Image.open(image_path).convert("RGB")
    if image_size is not None and min(image.size) > image_size:
        image: Image.Image = F.resize(
            image, image_size, interpolation=transforms.InterpolationMode.LANCZOS
        )

    original_size = image.size
    image = pad_to_divisible(image, divisor=16)
    # _transforms.append(transforms.CenterCrop((h // 16 * 16, w // 16 * 16)))
    image = F.to_tensor(image)
    return image.unsqueeze(0), original_size


def save_image(
    image: torch.Tensor, image_path: Union[str, Path], unpadded_size: tuple[int, int]
):
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    image = image.clamp(0, 1)
    image: Image.Image = F.to_pil_image(image)
    image = remove_pad(image, *unpadded_size)

    image.save(image_path)


def change_seg(seg):
    color_dict = {
        (0, 0, 255): 3,  # blue
        (0, 255, 0): 2,  # green
        (0, 0, 0): 0,  # black
        (255, 255, 255): 1,  # white
        (255, 0, 0): 4,  # red
        (255, 255, 0): 5,  # yellow
        (128, 128, 128): 6,  # grey
        (0, 255, 255): 7,  # lightblue
        (255, 0, 255): 8,  # purple
    }
    arr_seg = np.asarray(seg)
    new_seg = np.zeros(arr_seg.shape[:-1])
    for x in range(arr_seg.shape[0]):
        for y in range(arr_seg.shape[1]):
            if tuple(arr_seg[x, y, :]) in color_dict:
                new_seg[x, y] = color_dict[tuple(arr_seg[x, y, :])]
            else:
                min_dist_index = 0
                min_dist = 99999
                for key in color_dict:
                    dist = np.sum(np.abs(np.asarray(key) - arr_seg[x, y, :]))
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_index = color_dict[key]
                    elif dist == min_dist:
                        try:
                            min_dist_index = new_seg[x, y - 1, :]
                        except Exception:
                            pass
                new_seg[x, y] = min_dist_index
    return new_seg.astype(np.uint8)


def load_segment(image_path, image_size=None):
    if not image_path:
        return np.asarray([])
    image = Image.open(image_path)
    if image.mode == "RGBA":
        image = image.getchannel("A").convert("RGB")
    else:
        image = image.convert("RGB")

    if image_size is not None and min(image.size) > image_size:
        image: Image.Image = F.resize(
            image, image_size, interpolation=transforms.InterpolationMode.NEAREST
        )
    # w, h = image.size
    # transform = transforms.CenterCrop((h // 16 * 16, w // 16 * 16))
    # image = transform(image)
    image = pad_to_divisible(image, divisor=16)
    if len(np.asarray(image).shape) == 3:
        image = change_seg(image)
    return np.asarray(image)


def compute_label_info(content_segment, style_segment):
    if not content_segment.size or not style_segment.size:
        return None, None
    max_label = np.max(content_segment) + 1
    label_set = np.unique(content_segment)
    label_indicator = np.zeros(max_label)
    for l in label_set:
        content_mask = np.where(
            content_segment.reshape(content_segment.shape[0] * content_segment.shape[1])
            == l
        )
        style_mask = np.where(
            style_segment.reshape(style_segment.shape[0] * style_segment.shape[1]) == l
        )

        c_size = content_mask[0].size
        s_size = style_mask[0].size
        if (
            c_size > 10
            and s_size > 10
            and c_size / s_size < 100
            and s_size / c_size < 100
        ):
            label_indicator[l] = True
        else:
            label_indicator[l] = False
    return label_set, label_indicator


def mkdir(dname):
    if not os.path.exists(dname):
        os.makedirs(dname)
    else:
        assert os.path.isdir(dname), "alread exists filename {}".format(dname)


def collect_images(path: Union[str, Path]) -> list[Path]:
    mime_checker = mimetypes.MimeTypes()

    def validate_file_type(path: Path):
        mime_type = mime_checker.guess_type(path)[0]
        return mime_type is not None and mime_type.startswith("image")

    path = Path(path)
    if path.is_dir():
        return sorted(filter(validate_file_type, path.rglob("*.*")))
    else:
        return [path]


def collect_images_from_images(
    images: list[Union[str, Path]],
    root_path: Union[str, Path],
    target_path: Union[str, Path],
) -> list[Path]:
    target_path = Path(target_path)
    if target_path.is_file():
        return [target_path]

    paths = []
    exts = [
        ext
        for ext, mime_type in mimetypes.types_map.items()
        if mime_type.startswith("image")
    ]
    exts += [ext.upper() for ext in exts]
    for image in images:
        image = Path(image)
        rel_path = image.relative_to(root_path)

        for ext in exts:
            path = target_path / rel_path.with_suffix(ext)
            if path.exists():
                break
        else:
            assert (
                False
            ), f"The corresponding background image is not found, {rel_path}."

        paths.append(path)
    return paths
