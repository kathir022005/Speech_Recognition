# Speech Recognition Tool – Automatic Speech Recognition (ASR) Project

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![ASR Engine](https://img.shields.io/badge/ASR-OpenAI--Whisper-purple.svg)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready **Automatic Speech Recognition (ASR)** tool powered by **OpenAI Whisper**. The system transcribes audio files into text with support for **100+ languages**, multiple audio formats, real-time language detection, timestamped segments, and SRT subtitle export — all through a modern web interface, REST API, and command-line interface.

**Team Name:** KATHIRESAN S

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Project Objectives](#project-objectives)
3. [Key Features](#key-features)
4. [System Architecture](#system-architecture)
5. [ASR Workflow & Processing Pipeline](#asr-workflow--processing-pipeline)
6. [Technology Stack](#technology-stack)
7. [Project Structure](#project-structure)
8. [Installation & Setup](#installation--setup)
9. [How to Run](#how-to-run)
10. [Test Suite](#test-suite)
11. [API Reference](#api-reference)
12. [Screenshots & UI Walkthrough](#screenshots--ui-walkthrough)
13. [Future Enhancements](#future-enhancements)
14. [Contributors & License](#contributors--license)

---

## Problem Statement

Manual transcription of audio content is extremely time-consuming, error-prone, and inaccessible to non-specialists. Organizations, students, researchers, and content creators often need to convert speech recordings — lectures, interviews, meetings, podcasts — into accurate text. Existing commercial ASR APIs raise privacy concerns by routing sensitive audio through third-party cloud servers, and they incur per-minute usage costs.

The **Speech Recognition Tool** addresses these challenges by providing a fully **local, offline-capable** ASR solution using **OpenAI's Whisper** model. Users upload audio via a browser or command line, and the system returns accurate transcriptions with timestamps — without sending any data to external services.

---

## Project Objectives

* **Accurate Offline Transcription**: Utilize OpenAI Whisper for high-accuracy, privacy-preserving speech-to-text without any cloud API dependency.
* **Multi-Language Support**: Automatically detect and transcribe 100+ languages; optionally translate any language to English.
* **Multiple Audio Formats**: Accept WAV, MP3, FLAC, OGG, M4A, WebM, and MP4 audio files seamlessly.
* **Flexible Model Selection**: Choose from `tiny`, `base`, `small`, `medium`, or `large` Whisper models based on accuracy vs. speed tradeoff.
* **Timestamped Output**: Produce per-segment timestamps suitable for subtitle generation (SRT export supported).
* **Three Access Modes**: Provide Web UI, REST API, and CLI for maximum flexibility.
* **Comprehensive Testing**: Include automated unit tests for both the ASR engine and the Flask API.

---

## Key Features

### 1. Web-Based Audio Transcription
* Modern dark-themed drag-and-drop upload interface.
* Real-time model selection (tiny → large) from dropdown.
* Optional language specification or automatic detection.
* Transcribe or Translate (any language → English) task toggle.
* Results display with full text, metadata badges, and expandable timestamped segments table.

### 2. ASR Engine & Speech Processing
* Uses **OpenAI Whisper** transformer models with automatic device selection (CUDA GPU or CPU).
* Supports 5 model sizes: `tiny` (39M params), `base` (74M), `small` (244M), `medium` (769M), `large` (1550M).
* Automatic language detection with confidence scores and top-5 language probabilities.
* Audio preprocessing via `ffmpeg` backend for universal format decoding.

### 3. Smart Text Normalization & Output
* Full transcription text with leading/trailing whitespace stripped.
* Per-segment output with `id`, `start` timestamp, `end` timestamp, and segment `text`.
* Processing duration tracking for performance benchmarking.
* Multiple CLI output formats: `text`, `json`, `srt`, `verbose`.

### 4. REST API for Programmatic Access
* `POST /api/transcribe` — Upload audio file and receive JSON transcription.
* `POST /api/detect-language` — Detect spoken language with confidence scores.
* `GET /api/models` — List available models and supported formats.
* `GET /api/health` — Health check endpoint for monitoring.
* 100 MB max upload size with automatic temp-file cleanup.

### 5. Command-Line Interface
* `transcribe` command with batch file support (multiple files in one run).
* `detect` command for language identification.
* `info` command to display loaded model details.
* Output to file with `--output` flag.
* SRT subtitle export for video captioning workflows.

### 6. Comprehensive Test Suite
* **16 unit tests** for the ASR engine: initialization, validation, transcription, language detection.
* **8 unit tests** for the Flask app: endpoints, upload validation, mocked transcription API.
* Synthetic sine-wave WAV generation for reliable, microphone-free testing.
* Mocked engine tests for fast CI/CD pipeline integration.

---

## System Architecture

```text
+-----------------------------------------------------------------------+
|                             USER BROWSER                              |
|                                                                       |
|  [ File Upload ]  -->  Drag & Drop / File Picker UI                   |
|         |                         |                                   |
|  [ Model Select ]  [ Language ]   |  (POST /transcribe)               |
|  [ Task Toggle ]                  |  (POST /api/transcribe)           |
+-----------|-----------------------|-----------------------------------+
            |                       v
+-----------|-----------------------------------------------------------+
|           |              FLASK BACKEND (app.py)                       |
|           |                                                           |
|           |   1. Validate file format & size                          |
|           |   2. Save temp upload with UUID filename                  |
|           |   3. Load / reuse ASR Engine (lazy singleton)              |
|           |   4. Call engine.transcribe() or engine.detect_language()  |
|           |   5. Return JSON / rendered HTML result                   |
|           |   6. Cleanup temp file                                    |
|           |                                                           |
+-----------|-----------------------------------------------------------+
            |
            v
+-----------------------------------------------------------------------+
|                       ASR ENGINE (asr/engine.py)                      |
|                                                                       |
|   +-------------------+    +--------------------+                     |
|   | whisper.load_model|    | whisper.load_audio  |                    |
|   | (tiny/base/small/ |--->| pad_or_trim         |                    |
|   |  medium/large)    |    | log_mel_spectrogram |                    |
|   +-------------------+    +--------------------+                     |
|           |                         |                                 |
|           v                         v                                 |
|   +-------------------+    +--------------------+                     |
|   | model.transcribe()|    | model.detect_lang() |                    |
|   | - text             |    | - language code     |                    |
|   | - segments[]       |    | - probabilities{}   |                    |
|   | - language         |    +--------------------+                     |
|   +-------------------+                                               |
|                                                                       |
+-----------------------------------------------------------------------+
            |
            v
+-----------------------------------------------------------------------+
|                        CLI INTERFACE (cli.py)                         |
|                                                                       |
|   python cli.py transcribe audio.wav -m base -f json -o result.json   |
|   python cli.py detect audio.wav                                      |
|   python cli.py info                                                  |
+-----------------------------------------------------------------------+
```

---

## ASR Workflow & Processing Pipeline

```text
  Audio File (WAV/MP3/FLAC/OGG/M4A)
           │
           ▼
  ┌─────────────────────┐
  │  Format Validation  │  Check extension against SUPPORTED_FORMATS
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  FFmpeg Decoding     │  Whisper internally uses ffmpeg to decode
  │  → 16kHz mono float │  any format to raw 16kHz mono PCM
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  Log-Mel Spectrogram │  80-channel mel filterbank
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  Whisper Transformer │  Encoder-Decoder attention
  │  (chosen model size) │  with beam search decoding
  └─────────┬───────────┘
            │
            ├──► Full transcribed text
            ├──► Per-segment timestamps
            ├──► Detected language + confidence
            └──► Processing duration (seconds)
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **ASR Model** | [OpenAI Whisper](https://github.com/openai/whisper) | Transformer-based multilingual speech recognition |
| **Deep Learning** | [PyTorch](https://pytorch.org/) | Model inference engine (CUDA / CPU) |
| **Audio Decoding** | [FFmpeg](https://ffmpeg.org/) | Universal audio format decoding |
| **Audio Processing** | [Librosa](https://librosa.org/), [SoundFile](https://python-soundfile.readthedocs.io/) | Audio analysis and WAV I/O |
| **Backend Framework** | [Flask 3.x](https://flask.palletsprojects.com/) | Web server, routing, REST API |
| **Frontend** | HTML5, CSS3, JavaScript | Responsive dark-themed upload UI |
| **Testing** | [Pytest](https://docs.pytest.org/) | Unit and integration test framework |
| **Audio Utility** | [PyDub](https://github.com/jiaaro/pydub) | Audio segment manipulation |
| **Numerical** | [NumPy](https://numpy.org/) | Array operations for audio data |
| **Language** | Python 3.9+ | Core programming language |
| **Version Control** | [Git](https://git-scm.com/) / [GitHub](https://github.com/) | Source control and hosting |

---

## Project Structure

```
Speech_Recognition/
│
├── app.py                      # Flask web application (Web UI + REST API)
├── cli.py                      # Command-line interface for batch processing
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation (this file)
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
│
├── asr/                        # ASR engine package
│   ├── __init__.py             # Package exports
│   └── engine.py               # Core Whisper-based ASR engine
│
├── utils/                      # Utility modules
│   ├── __init__.py             # Package exports
│   └── audio_utils.py          # Audio validation & helper functions
│
├── templates/                  # Flask Jinja2 HTML templates
│   └── index.html              # Main web UI (dark theme, drag-and-drop)
│
├── static/                     # Static assets (CSS/JS if separated)
│   └── .gitkeep
│
├── uploads/                    # Temporary upload directory (auto-cleaned)
│   └── .gitkeep
│
└── tests/                      # Automated test suite
    ├── __init__.py
    ├── test_asr_engine.py      # Unit tests for ASR engine (16 tests)
    └── test_app.py             # Unit tests for Flask API (8 tests)
```

---

## Installation & Setup

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9+ | [Download](https://www.python.org/downloads/) |
| FFmpeg | Latest | Required by Whisper for audio decoding |
| Git | Latest | For cloning the repository |

### Install FFmpeg

**Windows (Chocolatey):**
```bash
choco install ffmpeg
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

### Clone & Install

```bash
# 1. Clone the repository
git clone https://github.com/kathir022005/Speech_Recognition.git
cd Speech_Recognition

# 2. Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** On first run, Whisper will automatically download the selected model weights (~75 MB for `tiny`, ~140 MB for `base`). This requires an internet connection once.

---

## How to Run

### Option 1: Web Interface

```bash
python app.py --model base --port 5000
```

Open your browser → **http://localhost:5000**

1. Drag & drop or select an audio file.
2. Choose model size, language (or auto-detect), and task.
3. Click **🚀 Transcribe Audio**.
4. View results with full text, language, timing, and expandable segments.

### Option 2: Command-Line Interface

```bash
# Transcribe a single file
python cli.py transcribe audio.wav

# Transcribe with specific model and JSON output
python cli.py -m small transcribe audio.mp3 --language en --output-format json --output result.json

# Batch transcribe multiple files
python cli.py transcribe file1.wav file2.mp3 file3.flac

# Generate SRT subtitles
python cli.py transcribe lecture.wav --output-format srt --output lecture.srt

# Detect language
python cli.py detect audio.wav

# Show model info
python cli.py info
```

### Option 3: REST API

```bash
# Transcribe audio via API
curl -X POST http://localhost:5000/api/transcribe \
  -F "audio_file=@audio.wav" \
  -F "model=base" \
  -F "language=en"

# Detect language
curl -X POST http://localhost:5000/api/detect-language \
  -F "audio_file=@audio.wav"

# List available models
curl http://localhost:5000/api/models

# Health check
curl http://localhost:5000/api/health
```

---

## Test Suite

### Running Tests

```bash
# Run full test suite
pytest tests/ -v

# Run only ASR engine tests
pytest tests/test_asr_engine.py -v

# Run only Flask app tests
pytest tests/test_app.py -v

# Run with coverage report
pytest tests/ -v --tb=short
```

### Test Coverage Summary

| Test File | Tests | Coverage Area |
|---|---|---|
| `test_asr_engine.py` | 16 | Engine init, model validation, transcription, language detection, format validation |
| `test_app.py` | 8 | Health check, models endpoint, upload validation, mocked transcription API |
| **Total** | **24** | Full engine + API coverage |

> **Note:** Engine tests require the `tiny` Whisper model (~75 MB, downloaded automatically on first run). App tests use mocked engines for fast execution.

---

## API Reference

### `POST /api/transcribe`

Transcribe an uploaded audio file to text.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio_file` | File | ✅ | — | Audio file (WAV, MP3, FLAC, OGG, M4A) |
| `model` | String | ❌ | `base` | Whisper model: tiny, base, small, medium, large |
| `language` | String | ❌ | auto-detect | ISO language code (e.g., `en`, `es`, `fr`, `ta`) |
| `task` | String | ❌ | `transcribe` | `transcribe` or `translate` (→ English) |

**Success Response (200):**
```json
{
  "success": true,
  "result": {
    "text": "Hello, how are you today?",
    "segments": [
      {"id": 0, "start": 0.0, "end": 2.48, "text": "Hello, how are you today?"}
    ],
    "language": "en",
    "duration": 1.23,
    "model": "base"
  },
  "timestamp": "2026-09-30T14:30:00.000000"
}
```

### `POST /api/detect-language`

Detect the spoken language in an audio file.

**Success Response (200):**
```json
{
  "success": true,
  "result": {
    "language": "en",
    "language_probability": 0.9876,
    "top_5_languages": {
      "en": 0.9876, "de": 0.0054, "fr": 0.0032, "es": 0.0018, "it": 0.0010
    }
  }
}
```

### `GET /api/models`

List available models and supported audio formats.

### `GET /api/health`

Health check endpoint. Returns `{"status": "healthy"}`.

---

## Screenshots & UI Walkthrough

### Main Upload Interface
The web interface features a dark gradient theme with a drag-and-drop upload zone, model/language/task selectors, and a prominent transcription button.

### Transcription Results
Results are displayed with metadata badges (filename, language, model, processing time), the full transcription text, and an expandable segments table with per-segment timestamps.

### CLI Output
The CLI provides verbose output with language, model info, processing time, full text, and timestamped segments — or clean JSON/SRT output for scripting.

---

## Future Enhancements

* **Real-Time Microphone Input**: Add browser-based microphone recording with Web Audio API for live transcription.
* **Speaker Diarization**: Integrate `pyannote-audio` to identify and label different speakers.
* **WebSocket Streaming**: Stream long audio files for real-time progressive transcription display.
* **Faster-Whisper Integration**: Add CTranslate2-based Faster-Whisper backend for 4× speed improvement.
* **Word-Level Timestamps**: Enable Whisper's word-level timestamp mode for precise subtitle alignment.
* **Batch Processing Dashboard**: Add a web-based batch upload queue with progress tracking.
* **Docker Container**: Provide a Docker image with GPU support for easy deployment.
* **Database Logging**: Log all transcription requests and results to SQLite for audit trails.

---

## Contributors & License

### Contributors

| Name | Role |
|---|---|
| **KATHIRESAN S** | Developer & Maintainer |

### License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

### Acknowledgements

* [OpenAI Whisper](https://github.com/openai/whisper) — State-of-the-art ASR model
* [Flask](https://flask.palletsprojects.com/) — Lightweight Python web framework
* [PyTorch](https://pytorch.org/) — Deep learning framework
* [FFmpeg](https://ffmpeg.org/) — Universal audio/video toolkit
