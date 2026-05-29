"""Single lazy EasyOCR reader shared by jersey digit fallback and name OCR."""

from __future__ import annotations

import os

_reader = None


def get_easyocr_reader():
    global _reader
    if _reader is None:
        import easyocr

        use_gpu = os.environ.get("OCR_GPU", "0") == "1"
        _reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    return _reader
