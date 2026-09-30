# 🎤 ASR Tool – Automatic Speech Recognition

An **Automatic Speech Recognition (ASR)** tool built with **OpenAI's Whisper** model. It converts speech from audio files into text with support for multiple languages, audio formats, and output styles.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Features](#-features)
- [Tools & Technologies](#-tools--technologies)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Web Interface](#1-web-interface)
  - [CLI](#2-command-line-interface)
  - [REST API](#3-rest-api)
- [Running Tests](#-running-tests)
- [API Reference](#-api-reference)
- [Screenshots](#-screenshots)
- [License](#-license)

---

## ✨ Features

- **Speech-to-Text Transcription** – Convert audio to text using state-of-the-art Whisper models
- **Multiple Model Sizes** – Choose from `tiny`, `base`, `small`, `medium`, or `large` based on accuracy/speed needs
- **Multi-format Support** – WAV, MP3, FLAC, OGG, M4A, WebM, MP4
- **Language Detection** – Automatically detect the spoken language
- **Translation** – Translate any language to English
- **Timestamped Segments** – Get word-level timestamps for subtitle generation
- **SRT Export** – Generate subtitle files in SRT format
- **Web UI** – Beautiful dark-themed web interface with drag-and-drop upload
- **REST API** – Programmatic access for integration with other tools
- **CLI** – Batch processing from the command line

---

## 🛠 Tools & Technologies

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core programming language |
| **OpenAI Whisper** | Speech recognition model (transformer-based) |
| **PyTorch** | Deep learning framework for model inference |
| **Flask** | Web framework for UI and REST API |
| **HTML/CSS/JS** | Frontend web interface |
| **Pytest** | Testing framework |
| **Librosa** | Audio processing and analysis |
| **NumPy** | Numerical computation |
| **Git/GitHub** | Version control and hosting |

---

## 📁 Project Structure

```
asr-tool/
├── app.py                  # Flask web application (UI + REST API)
├── asr_engine.py           # Core ASR engine (Whisper wrapper)
├── cli.py                  # Command-line interface
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
├── __init__.py             # Package metadata
├── templates/
│   └── index.html          # Web UI template
├── uploads/                # Temporary upload directory
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   ├── test_asr_engine.py  # Unit tests for ASR engine
│   └── test_app.py         # Unit tests for Flask app
└── sample_audio/           # Sample audio files for testing
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- `ffmpeg` installed on your system (required by Whisper)

#### Install ffmpeg

**Windows (using Chocolatey):**
```bash
choco install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/asr-tool.git
   cd asr-tool
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage

### 1. Web Interface

Start the Flask web server:

```bash
python app.py --model base --port 5000
```

Then open your browser and go to **http://localhost:5000**

You can:
- Upload an audio file via drag-and-drop or file picker
- Select the Whisper model size
- Choose the language or let it auto-detect
- View the full transcript with timestamps

### 2. Command-Line Interface

#### Transcribe an audio file:
```bash
python cli.py transcribe audio.wav
```

#### Transcribe with options:
```bash
python cli.py -m small transcribe audio.mp3 --language en --output-format json --output result.json
```

#### Detect the language:
```bash
python cli.py detect audio.wav
```

#### View model info:
```bash
python cli.py info
```

#### Output format options:
- `text` – Plain text only
- `json` – Structured JSON with segments
- `srt` – SRT subtitle format
- `verbose` – Full details with timestamps (default)

### 3. REST API

#### Transcribe audio:
```bash
curl -X POST http://localhost:5000/api/transcribe \
  -F "audio_file=@audio.wav" \
  -F "model=base" \
  -F "language=en"
```

#### Detect language:
```bash
curl -X POST http://localhost:5000/api/detect-language \
  -F "audio_file=@audio.wav"
```

#### List models:
```bash
curl http://localhost:5000/api/models
```

#### Health check:
```bash
curl http://localhost:5000/api/health
```

---

## 🧪 Running Tests

Run the full test suite:

```bash
pytest tests/ -v
```

Run only engine tests:
```bash
pytest tests/test_asr_engine.py -v
```

Run only app tests:
```bash
pytest tests/test_app.py -v
```

> **Note:** Some tests require the Whisper `tiny` model to be downloaded (~75MB on first run).

---

## 📡 API Reference

### `POST /api/transcribe`

Transcribe an uploaded audio file.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `audio_file` | File | Yes | Audio file to transcribe |
| `model` | String | No | Model size: tiny, base, small, medium, large (default: base) |
| `language` | String | No | Language code, e.g., `en`, `es` (default: auto-detect) |
| `task` | String | No | `transcribe` or `translate` (default: transcribe) |

**Response:**
```json
{
  "success": true,
  "result": {
    "text": "Hello, how are you?",
    "segments": [
      { "id": 0, "start": 0.0, "end": 2.5, "text": "Hello, how are you?" }
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

**Response:**
```json
{
  "success": true,
  "result": {
    "language": "en",
    "language_probability": 0.9876,
    "top_5_languages": {
      "en": 0.9876,
      "de": 0.0054,
      "fr": 0.0032,
      "es": 0.0018,
      "it": 0.0010
    }
  }
}
```

### `GET /api/models`

List available models and supported formats.

### `GET /api/health`

Health check endpoint.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) – The ASR model
- [Flask](https://flask.palletsprojects.com/) – Web framework
- [PyTorch](https://pytorch.org/) – Deep learning framework
