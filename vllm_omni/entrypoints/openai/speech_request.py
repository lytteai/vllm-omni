# SPDX-License-Identifier: Apache-2.0
"""Parse ``POST /v1/audio/speech`` from JSON or multipart form-data.

JSON keeps the existing OpenAI-compatible body (``ref_audio`` as a URL,
``data:`` URI, or ``file://`` URI). Multipart accepts the same fields as form
values and also lets clients upload ``ref_audio`` / ``ref_audio_2`` as files.
Uploaded bytes are converted to ``data:`` URIs so the rest of the speech stack
can keep using ``_resolve_ref_audio``.
"""

from __future__ import annotations

import base64
import json
from http import HTTPStatus
from numbers import Integral
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

SPEECH_REF_AUDIO_MAX_BYTES = 10 * 1024 * 1024

_AUDIO_EXT_TO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
}

_ALLOWED_AUDIO_MIME_TYPES = frozenset(
    {
        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/ogg",
        "audio/aac",
        "audio/flac",
        "audio/webm",
        "audio/mp4",
    }
)

_BOOL_FIELDS = frozenset(
    {
        "stream",
        "x_vector_only_mode",
        "non_streaming_mode",
        "word_timestamps",
    }
)
_JSON_FIELDS = frozenset({"extra_params", "speaker_embedding"})
_FILE_FIELDS = frozenset({"ref_audio", "ref_audio_2"})

_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})


def _media_type(raw_request: Request) -> str:
    header = raw_request.headers.get("content-type") or ""
    return header.split(";", 1)[0].strip().lower()


def _guess_audio_mime(filename: str | None, content_type: str | None) -> str:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime and mime not in {"application/octet-stream", "binary/octet-stream"}:
        return mime
    ext = Path(filename or "").suffix.lower()
    return _AUDIO_EXT_TO_MIME.get(ext, "audio/wav")


def _parse_form_bool(field: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise HTTPException(
        status_code=HTTPStatus.BAD_REQUEST.value,
        detail=f"Invalid boolean value for '{field}': {value}",
    )


async def _read_upload_limited(upload: UploadFile, *, max_bytes: int) -> bytes:
    declared_size = getattr(upload, "size", None)
    if isinstance(declared_size, Integral) and int(declared_size) > max_bytes:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"File size exceeds maximum limit of {max_bytes // (1024 * 1024)}MB.",
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(min(1024 * 1024, max_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"File size exceeds maximum limit of {max_bytes // (1024 * 1024)}MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _bytes_to_audio_data_url(content: bytes, *, filename: str | None, content_type: str | None) -> str:
    mime_type = _guess_audio_mime(filename, content_type)
    if mime_type not in _ALLOWED_AUDIO_MIME_TYPES:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"Unsupported MIME type: {mime_type}. Allowed: {sorted(_ALLOWED_AUDIO_MIME_TYPES)}",
        )
    audio_b64 = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{audio_b64}"


async def upload_to_audio_data_url(upload: UploadFile) -> str:
    """Read an uploaded audio file and return a ``data:`` URI."""
    content = await _read_upload_limited(upload, max_bytes=SPEECH_REF_AUDIO_MAX_BYTES)
    if not content:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="Uploaded audio file is empty")
    return _bytes_to_audio_data_url(content, filename=upload.filename, content_type=upload.content_type)


def _is_upload(value: Any) -> bool:
    return isinstance(value, UploadFile)


async def _resolve_form_audio_values(values: list[Any], *, field: str) -> str | list[str] | None:
    resolved: list[str] = []
    for value in values:
        if _is_upload(value):
            content = await _read_upload_limited(value, max_bytes=SPEECH_REF_AUDIO_MAX_BYTES)
            if not content:
                continue
            resolved.append(
                _bytes_to_audio_data_url(content, filename=value.filename, content_type=value.content_type)
            )
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                resolved.append(stripped)
            continue
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"'{field}' must be an audio file, URL, or data URI",
        )
    if not resolved:
        return None
    if len(resolved) == 1:
        return resolved[0]
    return resolved


def _coerce_form_value(key: str, value: str) -> Any:
    if key in _BOOL_FIELDS:
        return _parse_form_bool(key, value)
    if key in _JSON_FIELDS:
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"'{key}' must be valid JSON",
            ) from exc
    return value


async def _multipart_payload(raw_request: Request) -> dict[str, Any]:
    form = await raw_request.form()
    payload: dict[str, Any] = {}
    for key in form.keys():
        values = form.getlist(key)
        if key in _FILE_FIELDS:
            continue
        if any(_is_upload(value) for value in values):
            continue
        if not values:
            continue
        raw_value = values[0]
        if not isinstance(raw_value, str):
            continue
        if key != "input" and raw_value == "":
            continue
        payload[key] = _coerce_form_value(key, raw_value)

    ref_audio = await _resolve_form_audio_values(form.getlist("ref_audio"), field="ref_audio")
    if ref_audio is not None:
        payload["ref_audio"] = ref_audio
    ref_audio_2 = await _resolve_form_audio_values(form.getlist("ref_audio_2"), field="ref_audio_2")
    if ref_audio_2 is not None:
        if isinstance(ref_audio_2, list):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="'ref_audio_2' accepts a single audio file or URI",
            )
        payload["ref_audio_2"] = ref_audio_2
    return payload


async def parse_create_speech_request(raw_request: Request) -> OpenAICreateSpeechRequest:
    """Parse JSON or multipart ``/v1/audio/speech`` bodies into the speech schema."""
    media_type = _media_type(raw_request)
    if media_type == "multipart/form-data":
        payload = await _multipart_payload(raw_request)
    elif media_type in ("", "application/json"):
        try:
            payload = await raw_request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="Invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="JSON body must be an object")
    else:
        raise HTTPException(
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE.value,
            detail="Content-Type must be application/json or multipart/form-data",
        )
    try:
        return OpenAICreateSpeechRequest.model_validate(payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
