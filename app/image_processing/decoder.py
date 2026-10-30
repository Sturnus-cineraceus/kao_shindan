from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_DIMENSION = 10_000
SUPPORTED_FORMATS = {"JPEG", "PNG"}


class ImageDecodeError(ValueError):
    """Base error for an image that cannot safely be decoded."""


class UnsupportedImageFormatError(ImageDecodeError):
    pass


class EmptyImageError(ImageDecodeError):
    pass


class InvalidImageSizeError(ImageDecodeError):
    pass


@dataclass(frozen=True)
class DecodedImage:
    rgb: np.ndarray
    width: int
    height: int


def decode_image(data: bytes) -> DecodedImage:
    if not data:
        raise EmptyImageError("Image input is empty")
    if len(data) > MAX_INPUT_BYTES:
        raise InvalidImageSizeError("Image input exceeds the byte limit")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in SUPPORTED_FORMATS:
                raise UnsupportedImageFormatError("Only JPEG and PNG images are supported")
            if getattr(source, "n_frames", 1) != 1:
                raise UnsupportedImageFormatError("Animated or multi-frame images are not supported")
            width, height = source.size
            if (width < 1 or height < 1 or width > MAX_DIMENSION or height > MAX_DIMENSION
                    or width * height > MAX_IMAGE_PIXELS):
                raise InvalidImageSizeError("Image dimensions exceed the supported limit")
            source.load()
            oriented = ImageOps.exif_transpose(source)
            rgb_image = oriented.convert("RGB")
            rgb = np.array(rgb_image, dtype=np.uint8, copy=True)
            # Return only pixel data: no source image or metadata is retained.
            height, width = rgb.shape[:2]
            return DecodedImage(rgb=rgb, width=width, height=height)
    except ImageDecodeError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError("Image bytes could not be decoded") from exc
