from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.image_processing import decode_image
from app.image_processing.decoder import (
    EmptyImageError,
    ImageDecodeError,
    InvalidImageSizeError,
    UnsupportedImageFormatError,
)


def encoded(format: str, size: tuple[int, int] = (3, 2), **kwargs: object) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, (12, 34, 56)).save(stream, format=format, **kwargs)
    return stream.getvalue()


def test_decodes_jpeg_and_png_as_rgb_with_dimensions() -> None:
    for fmt in ("JPEG", "PNG"):
        result = decode_image(encoded(fmt))
        assert (result.width, result.height) == (3, 2)
        assert result.rgb.shape == (2, 3, 3)
        assert result.rgb.dtype == np.uint8


@pytest.mark.parametrize("data,error", [(b"", EmptyImageError), (b"not an image", ImageDecodeError)])
def test_rejects_empty_and_invalid_bytes(data: bytes, error: type[Exception]) -> None:
    with pytest.raises(error):
        decode_image(data)


def test_rejects_unsupported_image_format() -> None:
    with pytest.raises(UnsupportedImageFormatError):
        decode_image(encoded("BMP"))


def test_applies_exif_orientation_and_does_not_return_metadata() -> None:
    data = encoded("JPEG", (3, 2), exif=Image.Exif())
    image = Image.open(BytesIO(data))
    exif = image.getexif()
    exif[274] = 6
    stream = BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    result = decode_image(stream.getvalue())
    assert (result.width, result.height) == (2, 3)
    assert result.rgb.shape == (3, 2, 3)
    assert not hasattr(result, "exif")


def test_rejects_abnormal_dimensions() -> None:
    data = encoded("PNG", (10001, 1))
    with pytest.raises(InvalidImageSizeError):
        decode_image(data)
