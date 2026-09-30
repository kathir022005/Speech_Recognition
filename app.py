"""
app.py - Flask Web Application for ASR Tool
=============================================

Provides a web-based interface and REST API for the ASR tool.
Users can upload audio files through the browser or send requests
to the API endpoints for transcription.
"""

import os
import uuid
import logging
from datetime import datetime

from flask import Flask, render_template, request, jsonify, redirect, url_for

from asr_engine import ASREngine, SUPPORTED_FORMATS, AVAILABLE_MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max upload
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global ASR engine instance (lazy-loaded)
asr_engine = None


def get_engine(model_name: str = "base") -> ASREngine:
    """Get or create the ASR engine with the specified model."""
    global asr_engine
    if asr_engine is None or asr_engine.model_name != model_name:
        asr_engine = ASREngine(model_name=model_name)
    return asr_engine


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_FORMATS


# ──────────────────────────────────────────────
# Web Routes
# ──────────────────────────────────────────────


@app.route("/")
def index():
    """Render the main page."""
    return render_template(
        "index.html",
        models=AVAILABLE_MODELS,
        formats=sorted(SUPPORTED_FORMATS),
    )


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Handle audio file upload and transcription from the web form."""
    # Check if a file was uploaded
    if "audio_file" not in request.files:
        return render_template(
            "index.html",
            error="No file uploaded.",
            models=AVAILABLE_MODELS,
            formats=sorted(SUPPORTED_FORMATS),
        )

    file = request.files["audio_file"]
    if file.filename == "":
        return render_template(
            "index.html",
            error="No file selected.",
            models=AVAILABLE_MODELS,
            formats=sorted(SUPPORTED_FORMATS),
        )

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error=f"Unsupported file format. Supported: {sorted(SUPPORTED_FORMATS)}",
            models=AVAILABLE_MODELS,
            formats=sorted(SUPPORTED_FORMATS),
        )

    # Save the uploaded file
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    try:
        # Get parameters
        model_name = request.form.get("model", "base")
        language = request.form.get("language", "").strip() or None
        task = request.form.get("task", "transcribe")

        # Perform transcription
        engine = get_engine(model_name)
        result = engine.transcribe(
            audio_path=filepath,
            language=language,
            task=task,
        )

        return render_template(
            "index.html",
            result=result,
            filename=file.filename,
            models=AVAILABLE_MODELS,
            formats=sorted(SUPPORTED_FORMATS),
        )

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return render_template(
            "index.html",
            error=f"Transcription failed: {str(e)}",
            models=AVAILABLE_MODELS,
            formats=sorted(SUPPORTED_FORMATS),
        )

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


# ──────────────────────────────────────────────
# REST API Endpoints
# ──────────────────────────────────────────────


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    """
    REST API endpoint for transcription.

    Expects a multipart form with:
        - audio_file: The audio file to transcribe
        - model (optional): Whisper model size (default: 'base')
        - language (optional): Language code (default: auto-detect)
        - task (optional): 'transcribe' or 'translate' (default: 'transcribe')

    Returns:
        JSON response with transcription results.
    """
    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio_file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported format. Supported: {sorted(SUPPORTED_FORMATS)}"
        }), 400

    # Save file temporarily
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    try:
        model_name = request.form.get("model", "base")
        language = request.form.get("language", "").strip() or None
        task = request.form.get("task", "transcribe")

        engine = get_engine(model_name)
        result = engine.transcribe(
            audio_path=filepath,
            language=language,
            task=task,
        )

        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error(f"API transcription failed: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/detect-language", methods=["POST"])
def api_detect_language():
    """
    REST API endpoint for language detection.

    Expects a multipart form with:
        - audio_file: The audio file to analyze

    Returns:
        JSON response with detected language info.
    """
    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio_file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    try:
        model_name = request.form.get("model", "base")
        engine = get_engine(model_name)
        result = engine.detect_language(filepath)

        return jsonify({"success": True, "result": result})

    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/models", methods=["GET"])
def api_models():
    """Return available Whisper models and supported formats."""
    return jsonify({
        "models": AVAILABLE_MODELS,
        "supported_formats": sorted(SUPPORTED_FORMATS),
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    })


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR Tool Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--model", default="base", choices=AVAILABLE_MODELS,
                        help="Default Whisper model to load")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Pre-load the model
    logger.info(f"Pre-loading Whisper model: {args.model}")
    get_engine(args.model)

    app.run(host=args.host, port=args.port, debug=args.debug)
