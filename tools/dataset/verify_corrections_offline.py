"""Create correction clips and obtain an unbiased offline Whisper transcript."""

from __future__ import annotations

# --- repo layout shim: find shared modules (V5 batch core, dialogue/dataset tools, project_env) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT, _ROOT / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline", _ROOT / "tools" / "dataset", _ROOT / "tools" / "dialogue"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
# --- end shim ---
import argparse
import json
import re
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
PROJECT_ROOT = VERSION_DIR.parents[1]
DEFAULT_BENCHMARK_DIR = (
    VERSION_DIR
    / "audio_transcription_pilot"
    / "videos"
    / "6TnPRgY_loc"
    / "fresh_benchmark"
)
FFMPEG = (
    PROJECT_ROOT
    / "ffmpeg"
    / "ffmpeg-8.1.2-essentials_build"
    / "bin"
    / "ffmpeg.exe"
)
SEMANTIC_TYPES = {"proper_noun", "islamic_term", "organization_name"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_position(value: str) -> int:
    match = re.match(r"(\d+)", value)
    if not match:
        raise ValueError(f"Invalid word position: {value}")
    return int(match.group(1))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_pcm16_wav(path: Path):
    """Return mono 16-bit PCM samples as Whisper-compatible float32 audio."""
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit PCM WAV: {path}")
        if handle.getframerate() != 16000:
            raise ValueError(f"Expected 16 kHz WAV: {path}")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--model", default="small")
    parser.add_argument("--context-seconds", type=float, default=6.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    review = json.loads(
        (benchmark_dir / "correction_review.json").read_text(encoding="utf-8")
    )
    assembly = json.loads(
        (benchmark_dir / "assemblyai_response.json").read_text(encoding="utf-8")
    )
    audio_path = benchmark_dir.parent / "audio.mp3"
    if not FFMPEG.is_file() or not audio_path.is_file():
        print("ERROR: ffmpeg or source audio is missing")
        return 1

    semantic = [
        change for change in review["changes"] if change["type"] in SEMANTIC_TYPES
    ]
    words = assembly["words"]
    clips_dir = benchmark_dir / "audio_review_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, object]] = []
    for index, change in enumerate(semantic, start=1):
        position = first_position(change["position"])
        word = words[position - 1]
        center_seconds = (float(word["start"]) + float(word["end"])) / 2000.0
        start_seconds = max(0.0, center_seconds - args.context_seconds)
        duration_seconds = args.context_seconds * 2
        clip_path = clips_dir / f"{index:02d}_position_{change['position']}.wav"
        subprocess.run(
            [
                str(FFMPEG),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_seconds:.3f}",
                "-t",
                f"{duration_seconds:.3f}",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(clip_path),
            ],
            check=True,
        )
        prepared.append(
            {
                "position": change["position"],
                "source": change["source"],
                "corrected": change["corrected"],
                "clip_center_seconds": round(center_seconds, 3),
                "clip_start_seconds": round(start_seconds, 3),
                "clip_path": clip_path.relative_to(benchmark_dir).as_posix(),
            }
        )

    import whisper

    print(f"Loading offline Whisper {args.model} on CPU...", flush=True)
    model = whisper.load_model(args.model, device="cpu")
    for index, item in enumerate(prepared, start=1):
        print(f"Transcribing review clip {index}/{len(prepared)}...", flush=True)
        clip_audio = read_pcm16_wav(benchmark_dir / item["clip_path"])
        result = model.transcribe(
            clip_audio,
            language="en",
            task="transcribe",
            temperature=0,
            fp16=False,
            condition_on_previous_text=False,
            initial_prompt=None,
            verbose=False,
        )
        item["offline_whisper_text"] = result["text"].strip()
    output = {
        "generated_at": utc_now(),
        "purpose": "independent ASR corroboration; not a substitute for human listening",
        "network_or_paid_api_used": False,
        "model": f"openai-whisper-{args.model}",
        "model_prompted_with_expected_corrections": False,
        "source_audio": audio_path.relative_to(VERSION_DIR).as_posix(),
        "items": prepared,
    }
    write_json(benchmark_dir / "offline_audio_review.json", output)
    print(f"Review: {benchmark_dir / 'offline_audio_review.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
