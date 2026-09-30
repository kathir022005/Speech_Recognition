"""
utils/audio_utils.py - Audio Processing Utilities
===================================================

Helper routines for audio file validation, duration checks,
timestamp formatting, and SRT subtitle generation.
"""

import os
from typing import List, Dict

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".mp4"
}


def validate_audio_file(filepath: str) -> bool:
    """
    Validate that an audio file exists and has an accepted extension.

    Args:
        filepath: Full path to target audio file.

    Returns:
        bool: True if valid.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. Supported: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
        )
    return True


def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """
    Convert seconds float into HH:MM:SS.mmm or HH:MM:SS,mmm timestamp.

    Args:
        seconds: Float representation of elapsed seconds.
        srt_format: If True, uses comma as millisecond separator.

    Returns:
        Formatted timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    delimiter = "," if srt_format else "."
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", delimiter)


def generate_srt(segments: List[Dict]) -> str:
    """
    Generate SubRip (.srt) subtitle text from segments.

    Args:
        segments: List of segment dictionaries containing 'start', 'end', and 'text'.

    Returns:
        str: Formatted SRT content.
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start_ts = format_timestamp(seg["start"], srt_format=True)
        end_ts = format_timestamp(seg["end"], srt_format=True)
        text = seg.get("text", "").strip()

        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)
