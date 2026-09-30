"""
cli.py - Command-Line Interface for ASR Tool
==============================================

Provides a CLI for transcribing audio files directly
from the terminal without starting the web server.
"""

import argparse
import json
import sys
import os
import logging

from asr_engine import ASREngine, AVAILABLE_MODELS, SUPPORTED_FORMATS


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def cmd_transcribe(args) -> None:
    """Handle the transcribe command."""
    engine = ASREngine(model_name=args.model)

    for audio_path in args.audio_files:
        if not os.path.exists(audio_path):
            print(f"Error: File not found: {audio_path}", file=sys.stderr)
            continue

        print(f"\n{'='*60}")
        print(f"Transcribing: {audio_path}")
        print(f"{'='*60}")

        try:
            result = engine.transcribe(
                audio_path=audio_path,
                language=args.language,
                task=args.task,
            )

            if args.output_format == "json":
                print(json.dumps(result, indent=2, ensure_ascii=False))
            elif args.output_format == "text":
                print(f"\n{result['text']}\n")
            elif args.output_format == "srt":
                print_srt(result["segments"])
            elif args.output_format == "verbose":
                print(f"\nLanguage: {result['language']}")
                print(f"Model:    {result['model']}")
                print(f"Duration: {result['duration']}s")
                print(f"\nFull Text:\n{result['text']}\n")
                if result["segments"]:
                    print("Segments:")
                    for seg in result["segments"]:
                        start = format_timestamp(seg["start"])
                        end = format_timestamp(seg["end"])
                        print(f"  [{start} --> {end}] {seg['text']}")

            # Save to file if requested
            if args.output:
                output_path = args.output
                if len(args.audio_files) > 1:
                    base = os.path.splitext(os.path.basename(audio_path))[0]
                    ext = ".json" if args.output_format == "json" else ".txt"
                    output_path = os.path.join(
                        os.path.dirname(args.output), f"{base}{ext}"
                    )

                with open(output_path, "w", encoding="utf-8") as f:
                    if args.output_format == "json":
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    else:
                        f.write(result["text"])
                print(f"\nSaved to: {output_path}")

        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}", file=sys.stderr)


def cmd_detect(args) -> None:
    """Handle the detect-language command."""
    engine = ASREngine(model_name=args.model)

    result = engine.detect_language(args.audio_file)

    print(f"\nDetected Language: {result['language']}")
    print(f"Confidence: {result['language_probability']:.2%}")
    print(f"\nTop 5 Languages:")
    for lang, prob in result["top_5_languages"].items():
        bar = "█" * int(prob * 40)
        print(f"  {lang}: {prob:.4f} {bar}")


def cmd_info(args) -> None:
    """Handle the info command."""
    engine = ASREngine(model_name=args.model)
    info = engine.get_model_info()

    print(f"\nASR Tool - Model Information")
    print(f"{'='*40}")
    print(f"Model:             {info['model_name']}")
    print(f"Device:            {info['device']}")
    print(f"Parameters:        {info['parameters']}")
    print(f"Supported Formats: {', '.join(info['supported_formats'])}")
    print(f"Available Models:  {', '.join(info['available_models'])}")


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def print_srt(segments: list) -> None:
    """Print segments in SRT subtitle format."""
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start"]).replace(".", ",")
        end = format_timestamp(seg["end"]).replace(".", ",")
        print(f"{i}")
        print(f"{start} --> {end}")
        print(f"{seg['text']}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="asr-tool",
        description="ASR Tool - Automatic Speech Recognition using OpenAI Whisper",
    )
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=AVAILABLE_MODELS,
        help="Whisper model to use (default: base)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Transcribe command
    trans_parser = subparsers.add_parser(
        "transcribe", help="Transcribe audio file(s) to text"
    )
    trans_parser.add_argument(
        "audio_files", nargs="+", help="Audio file(s) to transcribe"
    )
    trans_parser.add_argument(
        "--language", "-l", default=None,
        help="Language code (e.g., 'en', 'es'). Auto-detects if not specified.",
    )
    trans_parser.add_argument(
        "--task", "-t", default="transcribe",
        choices=["transcribe", "translate"],
        help="Task to perform (default: transcribe)",
    )
    trans_parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path",
    )
    trans_parser.add_argument(
        "--output-format", "-f", default="verbose",
        choices=["text", "json", "srt", "verbose"],
        help="Output format (default: verbose)",
    )

    # Detect language command
    detect_parser = subparsers.add_parser(
        "detect", help="Detect the language of an audio file"
    )
    detect_parser.add_argument("audio_file", help="Audio file to analyze")

    # Info command
    subparsers.add_parser("info", help="Show model information")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "transcribe":
        cmd_transcribe(args)
    elif args.command == "detect":
        cmd_detect(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
