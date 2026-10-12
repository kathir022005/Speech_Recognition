"""
asr/engine.py - Core ASR Engine
================================

This module provides the core speech recognition functionality
using OpenAI's Whisper model. It handles model loading, audio
preprocessing, transcription, and language detection.
"""

import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies to allow server startup and testing
try:
    import whisper
    import torch
    import numpy as np
    WHISPER_AVAILABLE = True
except ImportError:
    whisper = None
    torch = None
    numpy = None
    WHISPER_AVAILABLE = False

# Supported audio file extensions
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".mp4"}

# Available Whisper model sizes
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]


class ASREngine:
    """
    Core Automatic Speech Recognition engine powered by OpenAI Whisper.

    This class handles:
    - Loading and managing Whisper models
    - Transcribing audio files to text
    - Detecting the spoken language
    - Providing transcription with timestamps

    Attributes:
        model_name (str): Name of the loaded Whisper model.
        device (str): Device used for inference ('cuda' or 'cpu').
        model: The loaded Whisper model instance.
    """

    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        """
        Initialize the ASR engine.

        Args:
            model_name: Whisper model size to use.
                        Options: 'tiny', 'base', 'small', 'medium', 'large'.
                        Defaults to 'base'.
            device: Device to run inference on ('cuda' or 'cpu').
                    If None, automatically selects CUDA if available.

        Raises:
            ValueError: If an invalid model name is provided.
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name '{model_name}'. "
                f"Available models: {AVAILABLE_MODELS}"
            )

        self.model_name = model_name
        if device:
            self.device = device
        elif torch is not None and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = None

        logger.info(
            f"Initializing ASR Engine with model='{model_name}' on device='{self.device}'"
        )
        self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model into memory."""
        if not WHISPER_AVAILABLE:
            raise RuntimeError(
                "OpenAI Whisper or PyTorch is not yet installed in the current Python environment. "
                "Please run `pip install -r requirements.txt` to install the speech recognition dependencies."
            )

        logger.info(f"Loading Whisper model '{self.model_name}'...")
        start_time = time.time()

        self.model = whisper.load_model(self.model_name, device=self.device)

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.2f}s")

    def _load_audio(self, audio_path: str):
        """
        Load audio file as a normalized float32 numpy array.
        Uses pure Python wave for WAV files or whisper.load_audio (ffmpeg) for other formats.
        """
        ext = os.path.splitext(audio_path)[1].lower()
        if ext == ".wav":
            try:
                import wave
                with wave.open(audio_path, "rb") as wf:
                    sr = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    frames = wf.readframes(wf.getnframes())

                    if sampwidth == 2:
                        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 4:
                        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                    elif sampwidth == 1:
                        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sampwidth}")

                    if n_channels > 1:
                        audio = audio.reshape(-1, n_channels).mean(axis=1)

                    # Resample to 16000Hz if needed
                    if sr != 16000 and len(audio) > 0:
                        target_len = int(len(audio) * 16000 / sr)
                        audio = np.interp(
                            np.linspace(0, len(audio), target_len, endpoint=False),
                            np.arange(len(audio)),
                            audio
                        ).astype(np.float32)

                    return audio
            except Exception as e:
                logger.warning(f"Native WAV decode fallback ({e}), attempting ffmpeg backend...")

        # Fallback to ffmpeg backend via whisper.load_audio
        try:
            return whisper.load_audio(audio_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to decode audio file '{os.path.basename(audio_path)}'. "
                f"For non-WAV formats, FFmpeg must be installed. Error: {e}"
            )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = False,
    ) -> dict:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file to transcribe.
            language: Language code (e.g., 'en', 'es', 'fr').
                      If None, language is auto-detected.
            task: Either 'transcribe' or 'translate' (to English).
            verbose: If True, print progress during transcription.

        Returns:
            dict: Transcription result containing:
                - 'text' (str): The full transcribed text.
                - 'segments' (list): List of segments with timestamps.
                - 'language' (str): Detected/specified language.
                - 'duration' (float): Processing time in seconds.

        Raises:
            FileNotFoundError: If the audio file doesn't exist.
            ValueError: If the file format is not supported.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{file_ext}'. "
                f"Supported formats: {SUPPORTED_FORMATS}"
            )

        if self.model is None:
            self._load_model()

        logger.info(f"Transcribing: {audio_path}")
        start_time = time.time()

        audio = self._load_audio(audio_path)

        options = {
            "task": task,
            "verbose": verbose,
        }
        if language:
            options["language"] = language

        result = self.model.transcribe(audio, **options)

        elapsed = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed:.2f}s")

        return {
            "text": result["text"].strip(),
            "segments": [
                {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
            "language": result.get("language", language or "unknown"),
            "duration": round(elapsed, 2),
            "model": self.model_name,
        }

    def detect_language(self, audio_path: str) -> dict:
        """
        Detect the language spoken in an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            dict: Detection result containing:
                - 'language' (str): Detected language code.
                - 'language_probability' (float): Confidence score.
                - 'all_probabilities' (dict): Top language probabilities.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.model is None:
            self._load_model()

        logger.info(f"Detecting language for: {audio_path}")

        audio = self._load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        _, probs = self.model.detect_language(mel)

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_language = sorted_probs[0]

        return {
            "language": top_language[0],
            "language_probability": round(top_language[1], 4),
            "top_5_languages": {
                lang: round(prob, 4) for lang, prob in sorted_probs[:5]
            },
        }

    def get_model_info(self) -> dict:
        """Get information about the currently loaded model."""
        num_params = sum(p.numel() for p in self.model.parameters()) if self.model else "Unloaded"
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": f"{num_params:,}" if isinstance(num_params, int) else num_params,
            "supported_formats": list(SUPPORTED_FORMATS),
            "available_models": AVAILABLE_MODELS,
        }


def create_engine(
    model_name: str = "base", device: Optional[str] = None
) -> ASREngine:
    """Factory function to create an ASR engine instance."""
    return ASREngine(model_name=model_name, device=device)
