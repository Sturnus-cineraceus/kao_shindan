from contextlib import asynccontextmanager
from dataclasses import asdict
from email import policy
from email.parser import BytesParser
import os
import logging
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from starlette.concurrency import run_in_threadpool

from app.analysis import AnalysisError, AnalysisService
from app.image_processing.decoder import MAX_INPUT_BYTES
from app.landmarks import FaceLandmarkerService
from app.logging_config import configure_logging
from app.statistics.models import validate_additional_features, ModelError

logger = configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    landmarker = FaceLandmarkerService(os.environ.get('FACE_LANDMARKER_MODEL_PATH', '/opt/kao-cal/models/face_landmarker.task'))
    app.state.analysis_service = AnalysisService(landmarker)
    try:
        yield
    finally:
        landmarker.close()


app = FastAPI(lifespan=lifespan)
web_dir = Path(__file__).resolve().parent
app.mount('/static', StaticFiles(directory=web_dir / 'static'), name='static')
templates = Jinja2Templates(directory=web_dir / 'templates')


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse(request=request, name='index.html', headers={'Cache-Control': 'no-store'})


def error(code: str, status: int, checks=()):
    issue_map = {
        'no_face': ('FACE_NOT_FOUND', '顔を検出できませんでした。正面を向いた顔が写る画像を選んでください。'),
        'multiple_faces': ('MULTIPLE_FACES', '複数の顔が検出されました。1人だけが写る画像を選んでください。'),
        'face_too_small': ('FACE_TOO_SMALL', '顔が小さすぎます。顔が大きく写る画像を選んでください。'),
        'landmark_near_edge': ('FACE_OUT_OF_FRAME', '顔が画像端に近すぎるか、必要な部分が画像外にあります。'),
        'yaw_out_of_range': ('HEAD_YAW_TOO_LARGE', '顔が横を向きすぎています。正面を向いた画像を選んでください。'),
        'pitch_out_of_range': ('HEAD_PITCH_TOO_LARGE', '顔が上下方向に向きすぎています。正面を向いた画像を選んでください。'),
        'roll_out_of_range': ('HEAD_ROLL_TOO_LARGE', '顔が傾きすぎています。まっすぐ撮影してください。'),
        'eye_closed': ('EYE_CLOSED', '目を開けた画像を選んでください。'),
        'mouth_open': ('MOUTH_OPEN', '口を閉じた画像を選んでください。'),
        'blurred': ('IMAGE_BLURRED', '画像がぼやけています。鮮明な画像を選んでください。'),
        'too_dark': ('IMAGE_TOO_DARK', '画像が暗すぎます。明るい場所で撮影してください。'),
        'too_bright': ('IMAGE_TOO_BRIGHT', '画像が明るすぎます。露出を調整してください。'),
        'shadow_clipping': ('BLACK_CLIPPING', '画像の暗い部分がつぶれています。'),
        'highlight_clipping': ('WHITE_CLIPPING', '画像の明るい部分が白飛びしています。'),
        'required_landmark_missing': ('REQUIRED_LANDMARK_MISSING', '瞳・目元・口元・顔輪郭の一部を確認できません。顔全体が写る画像を選んでください。'),
    }
    issues = []
    seen = set()
    for check in checks:
        mapped = issue_map.get(check.reason)
        if mapped and mapped[0] not in seen:
            seen.add(mapped[0])
            issues.append({'code': mapped[0], 'message': mapped[1]})
    message = issues[0]['message'] if issues else code
    return JSONResponse(status_code=status, content={'error': {'code': code, 'message': message, 'issues': issues}}, headers={'Cache-Control': 'no-store'})


@app.post('/analyze')
async def analyze(request: Request):
    origin = request.headers.get('origin')
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme != request.url.scheme or parsed.netloc != request.headers.get('host'):
            return error('invalid_request', 400)
    content_type = request.headers.get('content-type', '')
    if not content_type.lower().startswith('multipart/form-data;') or '\r' in content_type or '\n' in content_type:
        return error('invalid_request', 400)
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > MAX_INPUT_BYTES + 65536:
            return error('image_size_limit', 413)
        chunks.extend(chunk)
    message = BytesParser(policy=policy.default).parsebytes(
        b'Content-Type: ' + content_type.encode('latin-1') + b'\r\nMIME-Version: 1.0\r\n\r\n' + chunks)
    if not message.is_multipart() or message.defects:
        return error('invalid_request', 400)
    fields = {}
    additional_features = []
    for part in message.iter_parts():
        name = part.get_param('name', header='content-disposition')
        if name not in ('image', 'reference_group', 'additional_feature', 'calculation_profile') or (name != 'additional_feature' and name in fields) or part.is_multipart():
            return error('invalid_request', 400)
        if name == 'additional_feature':
            additional_features.append(part.get_payload(decode=True))
        else:
            fields[name] = part.get_payload(decode=True)
    if not {'image', 'reference_group'} <= set(fields) or any(value is None for value in fields.values()):
        return error('invalid_request', 400)
    try:
        group = fields['reference_group'].decode('utf-8')
    except UnicodeDecodeError:
        return error('invalid_reference_group', 400)
    try:
        profile = fields.get('calculation_profile', b'legacy').decode('utf-8')
    except UnicodeDecodeError:
        return error('invalid_calculation_profile', 400)
    if profile not in ('legacy', 'representative3'):
        return error('invalid_calculation_profile', 400)
    if profile == 'representative3' and additional_features:
        return error('features_not_allowed_for_profile', 400)
    try:
        selected = validate_additional_features(tuple(value.decode('utf-8') for value in additional_features))
    except (UnicodeDecodeError, AttributeError, ModelError):
        return error('invalid_feature_selection', 400)
    try:
        if profile == 'representative3':
            result = await run_in_threadpool(request.app.state.analysis_service.analyze, fields['image'], group, (), profile)
        elif selected:
            result = await run_in_threadpool(request.app.state.analysis_service.analyze, fields['image'], group, selected)
        else:
            result = await run_in_threadpool(request.app.state.analysis_service.analyze, fields['image'], group)
    except AnalysisError as exc:
        status = {'invalid_feature_selection': 400, 'invalid_reference_group': 400, 'invalid_image': 400, 'image_size_limit': 413,
                  'no_face': 422, 'multiple_faces': 422, 'validation_failed': 422,
                  'model_unavailable': 503, 'analysis_error': 500}.get(exc.code, 500)
        return error(exc.code, status, exc.checks)
    except Exception:
        logging.getLogger(__name__).exception('unexpected analysis API error')
        return error('analysis_error', 500)
    payload = asdict(result)
    # Keep the established success envelope while withholding internal check metrics.
    payload['quality_checks'] = [
        {'name': check.name, 'passed': check.passed, 'reason': check.reason}
        for check in result.quality_checks
    ]
    return JSONResponse(content=payload, headers={'Cache-Control': 'no-store'})
