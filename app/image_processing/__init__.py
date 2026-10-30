"""Decode untrusted image bytes into metadata-free RGB pixels."""

from .decoder import DecodedImage, ImageDecodeError, decode_image

__all__ = ["DecodedImage", "ImageDecodeError", "decode_image"]
