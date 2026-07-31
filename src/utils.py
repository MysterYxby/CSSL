from pathlib import Path

import cv2
import numpy as np
import torch


MAX_PIXEL_VALUE = 2047.0


def check_and_make(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def RSGenerate(image, percent, colorization=True):
    if not colorization:
        return image

    if image.ndim != 3:
        raise ValueError("image must have shape [height, width, channels]")

    height, width, channels = image.shape
    image_max = np.max(image)
    if image_max <= 0:
        return np.zeros_like(image, dtype=np.uint16)

    image_normalize = image / image_max
    image_generate = np.zeros_like(image_normalize)

    for channel in range(channels):
        image_slice = image_normalize[:, :, channel]
        pixels = np.sort(image_slice.reshape(height * width))
        high_index = np.floor(height * width * (1 - percent / 100)).astype(np.int32)
        low_index = np.ceil(height * width * percent / 100).astype(np.int32)
        high_index = np.clip(high_index, 0, len(pixels) - 1)
        low_index = np.clip(low_index, 0, len(pixels) - 1)
        maximum = pixels[high_index]
        minimum = pixels[low_index]
        image_generate[:, :, channel] = (image_slice - minimum) / (maximum - minimum + 1e-9)

    image_generate = np.clip(image_generate, 0, 1)
    image_generate = cv2.normalize(
        image_generate,
        dst=None,
        alpha=0,
        beta=65535,
        norm_type=cv2.NORM_MINMAX,
    )
    return image_generate.astype(np.uint16)


def save_image(ms_image, save_path, bands, flag_cut_bounds=False, dim_cut=0, ratio=4):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if flag_cut_bounds:
        width, height, _ = ms_image.shape
        crop = round(dim_cut / ratio)
        ms_image = ms_image[crop:-crop, crop:-crop, :]

    selected_bands = ms_image[:, :, bands]
    image = RSGenerate(selected_bands, percent=1, colorization=True)

    if flag_cut_bounds:
        image = cv2.resize(image, (width, height))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), image)


def TensorToImage(image_tensor, max_value=MAX_PIXEL_VALUE):
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor must be a PyTorch tensor")

    image_tensor = image_tensor.detach().cpu() * max_value
    channels, height, width = image_tensor.size(-3), image_tensor.size(-2), image_tensor.size(-1)

    if channels == 1:
        return image_tensor.reshape(height, width).numpy()

    image = np.zeros((height, width, channels), dtype=np.float32)
    for channel in range(channels):
        image[:, :, channel] = image_tensor[channel].numpy()
    return image
