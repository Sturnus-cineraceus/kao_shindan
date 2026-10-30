from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DESTINATION = Path(os.environ.get("FACE_LANDMARKER_MODEL_PATH", "/opt/kao-cal/models/face_landmarker.task"))


def main() -> None:
    if DESTINATION.is_file() and hashlib.sha256(DESTINATION.read_bytes()).hexdigest() == SHA256:
        return
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=DESTINATION.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        urllib.request.urlretrieve(URL, temp_path)
        digest = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        if digest != SHA256:
            raise RuntimeError(f"Model checksum mismatch: {digest}")
        temp_path.replace(DESTINATION)
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
