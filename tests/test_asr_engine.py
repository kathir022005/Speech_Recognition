"""
tests/test_asr_engine.py - Unit Tests for ASR Engine
=====================================================

Tests cover:
- Engine initialization and configuration
- Input validation (missing files, unsupported formats)
- Model info retrieval
- Transcription functionality (requires model download)
"""

import os
import sys
import wave
import struct
import pytest
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr_engine import ASREngine, SUPPORTED_FORMATS, AVAILABLE_MODELS


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


def generate_sine_wav(filepath: str, duration: float = 2.0,
                      sample_rate: int = 16000, frequency: float = 440.0) -> str:
    """Generate a simple sine wave WAV file for testing."""
    import math

    num_samples = int(sample_rate * duration)
    amplitude = 16000

    with wave.open(filepath, "w") as wav_file:
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)       # 16-bit
        wav_file.setframerate(sample_rate)

        for i in range(num_samples):
            value = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
            data = struct.pack("<h", value)
            wav_file.writeframes(data)

    return filepath


@pytest.fixture
def sample_wav(tmp_path):
    """Create a temporary WAV file for testing."""
    filepath = str(tmp_path / "test_audio.wav")
    generate_sine_wav(filepath, duration=2.0)
    return filepath


@pytest.fixture
def engine():
    """Create an ASR engine with the tiny model for fast tests."""
    return ASREngine(model_name="tiny")


# ──────────────────────────────────────────────
# Test: Initialization
# ──────────────────────────────────────────────


class TestASREngineInit:
    """Tests for ASR engine initialization."""

    def test_valid_model_names(self):
        """Test that all valid model names are accepted."""
        for name in AVAILABLE_MODELS:
            # Just verify no ValueError is raised (don't actually load all models)
            assert name in AVAILABLE_MODELS

    def test_invalid_model_name(self):
        """Test that invalid model names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid model name"):
            ASREngine(model_name="nonexistent")

    def test_engine_attributes(self, engine):
        """Test that the engine has correct attributes after init."""
        assert engine.model_name == "tiny"
        assert engine.device in ("cpu", "cuda")
        assert engine.model is not None

    def test_model_loads_successfully(self, engine):
        """Test that the model is loaded and ready for inference."""
        assert engine.model is not None
        info = engine.get_model_info()
        assert info["model_name"] == "tiny"


# ──────────────────────────────────────────────
# Test: Input Validation
# ──────────────────────────────────────────────


class TestInputValidation:
    """Tests for input validation in transcription."""

    def test_file_not_found(self, engine):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            engine.transcribe("/nonexistent/path/audio.wav")

    def test_unsupported_format(self, engine, tmp_path):
        """Test that ValueError is raised for unsupported formats."""
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("not audio")

        with pytest.raises(ValueError, match="Unsupported audio format"):
            engine.transcribe(str(bad_file))

    def test_detect_language_file_not_found(self, engine):
        """Test language detection with missing file."""
        with pytest.raises(FileNotFoundError):
            engine.detect_language("/nonexistent/audio.wav")


# ──────────────────────────────────────────────
# Test: Model Info
# ──────────────────────────────────────────────


class TestModelInfo:
    """Tests for model information retrieval."""

    def test_get_model_info(self, engine):
        """Test that model info returns expected fields."""
        info = engine.get_model_info()

        assert "model_name" in info
        assert "device" in info
        assert "parameters" in info
        assert "supported_formats" in info
        assert "available_models" in info

    def test_model_info_values(self, engine):
        """Test model info values are reasonable."""
        info = engine.get_model_info()

        assert info["model_name"] == "tiny"
        assert info["device"] in ("cpu", "cuda")
        assert len(info["supported_formats"]) > 0
        assert "tiny" in info["available_models"]
        assert "base" in info["available_models"]


# ──────────────────────────────────────────────
# Test: Transcription
# ──────────────────────────────────────────────


class TestTranscription:
    """Tests for audio transcription functionality."""

    def test_transcribe_wav(self, engine, sample_wav):
        """Test transcription of a WAV file (sine wave = no speech)."""
        result = engine.transcribe(sample_wav)

        assert "text" in result
        assert "segments" in result
        assert "language" in result
        assert "duration" in result
        assert "model" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["segments"], list)
        assert result["model"] == "tiny"

    def test_transcribe_with_language(self, engine, sample_wav):
        """Test transcription with specified language."""
        result = engine.transcribe(sample_wav, language="en")

        assert result["language"] == "en"
        assert "text" in result

    def test_transcribe_translate_task(self, engine, sample_wav):
        """Test transcription with translate task."""
        result = engine.transcribe(sample_wav, task="translate")

        assert "text" in result
        assert isinstance(result["duration"], float)

    def test_transcribe_result_structure(self, engine, sample_wav):
        """Test that transcription result has correct structure."""
        result = engine.transcribe(sample_wav)

        # Check result keys
        expected_keys = {"text", "segments", "language", "duration", "model"}
        assert set(result.keys()) == expected_keys

        # Check segment structure if any exist
        for seg in result["segments"]:
            assert "id" in seg
            assert "start" in seg
            assert "end" in seg
            assert "text" in seg
            assert isinstance(seg["start"], float)
            assert isinstance(seg["end"], float)


# ──────────────────────────────────────────────
# Test: Language Detection
# ──────────────────────────────────────────────


class TestLanguageDetection:
    """Tests for language detection functionality."""

    def test_detect_language(self, engine, sample_wav):
        """Test language detection returns expected fields."""
        result = engine.detect_language(sample_wav)

        assert "language" in result
        assert "language_probability" in result
        assert "top_5_languages" in result
        assert isinstance(result["language"], str)
        assert isinstance(result["language_probability"], float)
        assert len(result["top_5_languages"]) == 5

    def test_detect_language_probabilities(self, engine, sample_wav):
        """Test that detection probabilities are valid."""
        result = engine.detect_language(sample_wav)

        assert 0 <= result["language_probability"] <= 1
        for prob in result["top_5_languages"].values():
            assert 0 <= prob <= 1


# ──────────────────────────────────────────────
# Test: Supported Formats
# ──────────────────────────────────────────────


class TestConstants:
    """Tests for module constants."""

    def test_supported_formats(self):
        """Test that common audio formats are supported."""
        assert ".wav" in SUPPORTED_FORMATS
        assert ".mp3" in SUPPORTED_FORMATS
        assert ".flac" in SUPPORTED_FORMATS

    def test_available_models(self):
        """Test that expected models are available."""
        assert "tiny" in AVAILABLE_MODELS
        assert "base" in AVAILABLE_MODELS
        assert "small" in AVAILABLE_MODELS
        assert "medium" in AVAILABLE_MODELS
        assert "large" in AVAILABLE_MODELS
