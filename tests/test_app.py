"""
tests/test_app.py - Unit Tests for Flask Web Application
=========================================================

Tests cover:
- Health check endpoint
- API models endpoint
- File upload validation
- Transcription API (mocked)
"""

import os
import sys
import io
import json
import wave
import struct
import math
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_wav_bytes():
    """Generate a small WAV file as bytes for upload testing."""
    buf = io.BytesIO()
    sample_rate = 16000
    duration = 0.5
    num_samples = int(sample_rate * duration)

    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(num_samples):
            value = int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            wav_file.writeframes(struct.pack("<h", value))

    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────
# Test: Basic Endpoints
# ──────────────────────────────────────────────


class TestBasicEndpoints:
    """Tests for basic API endpoints."""

    def test_index_page(self, client):
        """Test that the index page loads."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"ASR Tool" in response.data

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_models_endpoint(self, client):
        """Test the models listing endpoint."""
        response = client.get("/api/models")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "models" in data
        assert "supported_formats" in data
        assert "base" in data["models"]
        assert ".wav" in data["supported_formats"]


# ──────────────────────────────────────────────
# Test: Upload Validation
# ──────────────────────────────────────────────


class TestUploadValidation:
    """Tests for file upload validation."""

    def test_no_file_api(self, client):
        """Test API response when no file is provided."""
        response = client.post("/api/transcribe")
        assert response.status_code == 400

        data = json.loads(response.data)
        assert "error" in data

    def test_empty_filename_api(self, client):
        """Test API response with empty filename."""
        response = client.post(
            "/api/transcribe",
            data={"audio_file": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_unsupported_format_api(self, client):
        """Test API response with unsupported file format."""
        response = client.post(
            "/api/transcribe",
            data={"audio_file": (io.BytesIO(b"data"), "test.xyz")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Unsupported" in data["error"]


# ──────────────────────────────────────────────
# Test: Transcription API (Mocked)
# ──────────────────────────────────────────────


class TestTranscriptionAPI:
    """Tests for the transcription API with mocked engine."""

    @patch("app.get_engine")
    def test_transcribe_success(self, mock_get_engine, client, sample_wav_bytes):
        """Test successful transcription via API."""
        # Mock the engine
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = {
            "text": "Hello world",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello world"}
            ],
            "language": "en",
            "duration": 0.5,
            "model": "base",
        }
        mock_get_engine.return_value = mock_engine

        response = client.post(
            "/api/transcribe",
            data={
                "audio_file": (io.BytesIO(sample_wav_bytes), "test.wav"),
                "model": "base",
                "language": "en",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["result"]["text"] == "Hello world"
        assert data["result"]["language"] == "en"

    @patch("app.get_engine")
    def test_detect_language_no_file(self, mock_get_engine, client):
        """Test language detection without file."""
        response = client.post("/api/detect-language")
        assert response.status_code == 400
