from contextlib import asynccontextmanager
from dataclasses import asdict
from email import policy
from email.parser import BytesParser
import os
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
    return JSONResponse(status_code=status, content={'error': {'code': code, 'message': code,
        'checks': [asdict(check) for check in checks]}}, headers={'Cache-Control': 'no-store'})


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
    for part in message.iter_parts():
        name = part.get_param('name', header='content-disposition')
        if name not in ('image', 'reference_group') or name in fields or part.is_multipart():
            return error('invalid_request', 400)
        fields[name] = part.get_payload(decode=True)
    if set(fields) != {'image', 'reference_group'} or any(value is None for value in fields.values()):
        return error('invalid_request', 400)
    try:
        group = fields['reference_group'].decode('utf-8')
    except UnicodeDecodeError:
        return error('invalid_reference_group', 400)
    try:
        result = await run_in_threadpool(request.app.state.analysis_service.analyze, fields['image'], group)
    except AnalysisError as exc:
        status = {'invalid_reference_group': 400, 'invalid_image': 400, 'image_size_limit': 413,
                  'no_face': 422, 'multiple_faces': 422, 'validation_failed': 422,
                  'model_unavailable': 503, 'analysis_error': 500}.get(exc.code, 500)
        return error(exc.code, status, exc.checks)
    except Exception:
        return error('analysis_error', 500)
    return JSONResponse(content=asdict(result), headers={'Cache-Control': 'no-store'})
