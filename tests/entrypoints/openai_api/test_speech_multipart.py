# SPDX-License-Identifier: Apache-2.0
"""L1 tests for JSON and multipart parsing of POST /v1/audio/speech."""

from __future__ import annotations

import base64
import io
import json
import wave

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.openai import api_server as api_server_module
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.speech_request import (
    SPEECH_REF_AUDIO_MAX_BYTES,
    parse_create_speech_request,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _wav_bytes(*, frames: int = 24000, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


@pytest.fixture
def parse_client() -> TestClient:
    app = FastAPI()

    @app.post("/parse")
    async def parse(raw_request: Request):
        request = await parse_create_speech_request(raw_request)
        return request.model_dump(mode="json")

    return TestClient(app)


def test_json_speech_request_still_works(parse_client: TestClient) -> None:
    response = parse_client.post(
        "/parse",
        json={
            "input": "Hello, this is a cloned voice.",
            "model": "fishaudio/s2-pro",
            "voice": "default",
            "ref_audio": "https://example.com/reference.wav",
            "ref_text": "Transcript of the reference audio.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["input"] == "Hello, this is a cloned voice."
    assert body["ref_audio"] == "https://example.com/reference.wav"
    assert body["ref_text"] == "Transcript of the reference audio."


def test_multipart_uploads_ref_audio_file(parse_client: TestClient) -> None:
    wav_bytes = _wav_bytes()
    response = parse_client.post(
        "/parse",
        data={
            "model": "fishaudio/s2-pro",
            "input": "Hello, this is a cloned voice.",
            "voice": "default",
            "ref_text": "Hey, why are you sitting here all alone in the corner?",
        },
        files={"ref_audio": ("reference.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["input"] == "Hello, this is a cloned voice."
    assert body["model"] == "fishaudio/s2-pro"
    assert body["voice"] == "default"
    assert body["ref_text"].startswith("Hey, why are you sitting here")
    assert body["ref_audio"].startswith("data:audio/wav;base64,")
    decoded = base64.b64decode(body["ref_audio"].split(",", 1)[1])
    assert decoded == wav_bytes


def test_multipart_ref_audio_url_string(parse_client: TestClient) -> None:
    response = parse_client.post(
        "/parse",
        data={
            "input": "Hello",
            "ref_audio": "https://example.com/reference.wav",
            "ref_text": "transcript",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["ref_audio"] == "https://example.com/reference.wav"


def test_multipart_rejects_unsupported_ref_audio_type(parse_client: TestClient) -> None:
    response = parse_client.post(
        "/parse",
        data={"input": "Hello"},
        files={"ref_audio": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported MIME type" in response.json()["detail"]


def test_multipart_rejects_oversized_ref_audio(parse_client: TestClient) -> None:
    too_big = b"a" * (SPEECH_REF_AUDIO_MAX_BYTES + 1)
    response = parse_client.post(
        "/parse",
        data={"input": "Hello"},
        files={"ref_audio": ("reference.wav", too_big, "audio/wav")},
    )
    assert response.status_code == 400
    assert "10MB" in response.json()["detail"]


def test_multipart_parses_bool_and_json_fields(parse_client: TestClient) -> None:
    response = parse_client.post(
        "/parse",
        data={
            "input": "Hello",
            "stream": "true",
            "response_format": "pcm",
            "extra_params": json.dumps({"temperature": 0.8}),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stream"] is True
    assert body["extra_params"] == {"temperature": 0.8}


def test_create_speech_multipart_forwards_uploaded_ref_audio(mocker: MockerFixture) -> None:
    handler = mocker.MagicMock()
    handler.create_speech = mocker.AsyncMock(return_value=Response(content=b"wav-bytes", media_type="audio/wav"))

    app = FastAPI()
    app.state.openai_serving_speech = handler
    app.add_api_route("/v1/audio/speech", api_server_module.create_speech, methods=["POST"])
    client = TestClient(app)

    wav_bytes = _wav_bytes()
    response = client.post(
        "/v1/audio/speech",
        data={
            "model": "fishaudio/s2-pro",
            "input": "Hello, this is a cloned voice.",
            "voice": "default",
            "ref_text": "Transcript of the reference audio.",
        },
        files={"ref_audio": ("reference.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    assert response.content == b"wav-bytes"

    called_request = handler.create_speech.call_args.args[0]
    assert isinstance(called_request, OpenAICreateSpeechRequest)
    assert called_request.input == "Hello, this is a cloned voice."
    assert called_request.ref_audio.startswith("data:audio/wav;base64,")
    assert called_request.ref_text == "Transcript of the reference audio."
