"""
asr_engine.py - Core ASR Engine
================================

This module provides the core speech recognition functionality
using OpenAI's Whisper model. It handles model loading, audio
preprocessing, transcription, and language detection.
"""

import os
import time
import logging
from typing import Optional

import whisper
import torch
import numpy as np

logger = logging.getLogger(__name__)

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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        logger.info(
            f"Initializing ASR Engine with model='{model_name}' on device='{self.device}'"
        )
        self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model into memory."""
        logger.info(f"Loading Whisper model '{self.model_name}'...")
        start_time = time.time()

        self.model = whisper.load_model(self.model_name, device=self.device)

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.2f}s")

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
        # Validate the input file
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{file_ext}'. "
                f"Supported formats: {SUPPORTED_FORMATS}"
            )

        logger.info(f"Transcribing: {audio_path}")
        start_time = time.time()

        # Build transcription options
        options = {
            "task": task,
            "verbose": verbose,
        }
        if language:
            options["language"] = language

        # Perform transcription
        result = self.model.transcribe(audio_path, **options)

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

        logger.info(f"Detecting language for: {audio_path}")

        # Load and pad/trim audio to 30 seconds
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # Create mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        # Detect language
        _, probs = self.model.detect_language(mel)

        # Sort probabilities and get top 5
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
        """
        Get information about the currently loaded model.

        Returns:
            dict: Model information including name, device,
                  and parameter count.
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": f"{num_params:,}",
            "supported_formats": list(SUPPORTED_FORMATS),
            "available_models": AVAILABLE_MODELS,
        }


def create_engine(
    model_name: str = "base", device: Optional[str] = None
) -> ASREngine:
    """
    Factory function to create an ASR engine instance.

    Args:
        model_name: Whisper model size. Default is 'base'.
        device: Compute device. Default is auto-detect.

    Returns:
        ASREngine: Configured ASR engine instance.
    """
    return ASREngine(model_name=model_name, device=device)
