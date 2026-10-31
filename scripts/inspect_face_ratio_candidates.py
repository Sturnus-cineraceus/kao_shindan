"""Compare provisional face-width and face-height proxy candidates on an image."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

# Support both direct execution and python -m scripts.inspect_face_ratio_candidates.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from app.feature_extraction.service import pixel_distance
from app.image_processing import decode_image
from app.landmarks import FaceLandmarkerService
from app.validation import validate

WIDTHS = {"width_candidate_234_454": (234, 454), "width_candidate_127_356": (127, 356)}
HEIGHTS = {"height_candidate_168_152": (168, 152), "height_candidate_6_152": (6, 152)}
PAIRS = {**WIDTHS, **HEIGHTS}


def measure_candidates(points, width, height):
    """Return positive finite pixel distances and height/width ratios."""
    if width <= 0 or height <= 0 or not math.isfinite(width) or not math.isfinite(height):
        raise ValueError("image dimensions must be finite and positive")
    distances = {}
    for name, (a, b) in PAIRS.items():
        if len(points) <= max(a, b):
            raise ValueError(f"required candidate landmark missing: {max(a, b)}")
        p, q = points[a], points[b]
        if not all(math.isfinite(v) for v in (p.x, p.y, q.x, q.y)):
            raise ValueError(f"{name}: non-finite coordinate")
        value = pixel_distance(p.x, p.y, q.x, q.y, width, height)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}: distance must be finite and positive")
        distances[name] = value
    ratios = {}
    for h in HEIGHTS:
        for w in WIDTHS:
            value = distances[h] / distances[w]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{h} / {w}: ratio must be finite and positive")
            ratios[(h, w)] = value
    return distances, ratios


def draw_candidates(image, points, output):
    # Draw on the original oriented image using original coordinates; rotation
    # preserves the pixel distances measured with the corrected coordinates.
    canvas = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2BGR)
    scale = max(0.45, min(image.width, image.height) / 1000)
    for index, (name, (a, b)) in enumerate(PAIRS.items()):
        color = (0, 200, 255) if name.startswith("width") else (255, 130, 0)
        coords = [tuple(round(v) for v in (points[i].x * image.width, points[i].y * image.height)) for i in (a, b)]
        cv2.line(canvas, *coords, color, 2, cv2.LINE_AA)
        for i, point in zip((a, b), coords):
            cv2.circle(canvas, point, 5, color, -1)
            cv2.putText(canvas, str(i), (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, name, (10, 25 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode('.png', canvas)
    if not success:
        raise ValueError('could not encode diagnostic image')
    encoded.tofile(str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--model', type=Path, default=Path('/opt/kao-cal/models/face_landmarker.task'))
    parser.add_argument('--output', type=Path, help='diagnostic PNG (default: tmp/face-ratio-diagnostics/<image>.png)')
    args = parser.parse_args()
    output = args.output or Path('tmp/face-ratio-diagnostics') / (args.image.stem + '.png')
    try:
        image = decode_image(args.image.read_bytes())
        service = FaceLandmarkerService(args.model)
        try:
            landmarks = service.analyze(image)
        finally:
            service.close()
        quality = validate(image, landmarks)
        print(f'image: {args.image.name}\nsize: {image.width} x {image.height}')
        print(f'validation: {"passed" if quality.passed else "failed"}')
        for check in quality.checks:
            print(f'  {check.name}: {"pass" if check.passed else "fail (" + str(check.reason) + ")"}')
        corrected = quality.analysis_landmarks
        roll_applied = corrected is not None and landmarks.face_count == 1 and corrected is not landmarks.faces[0]
        print(f'roll correction: {"applied" if roll_applied else "not applied"}')
        if not quality.passed or corrected is None:
            raise ValueError('validation failed; candidate measurements unavailable')
        distances, ratios = measure_candidates(corrected, image.width, image.height)
        for name, value in distances.items():
            print(f'{name} {PAIRS[name]}: {value:.2f} px')
        for (h, w), value in ratios.items():
            print(f'ratio {h} / {w}: {value:.4f}')
        draw_candidates(image, landmarks.faces[0], output)
        print(f'visualization: {output}')
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
