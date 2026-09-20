"""Two-stage Islamic/Arabic ASR correction pipeline using OpenRouter.

Call 1 detects likely ASR errors in independently processed transcript chunks.
Call 2 applies the detected corrections to the same chunks.  The YouTube
transcript is never supplied to either model and is read only after correction,
when the optional evaluation report is produced.

Preview prompts without making API calls:
    python two_call_pipeline.py --preview-prompts

Run the pipeline after approving the prompts:
    python two_call_pipeline.py
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = VERSION_DIR.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from project_env import load_env_file, require_env


VERSIONS_DIR = PROJECT_DIR / "transcription_pipeline_versions"
RESULTS_DIR = VERSION_DIR
SOURCE_PATH = PROJECT_DIR / "assemblyai_transcript.txt"
REFERENCE_PATH: Path | None = (
    VERSIONS_DIR
    / "v1_transcription_tool_comparison"
    / "yt_transcript_clean.txt"
)
ENV_PATH = PROJECT_DIR / ".env"

CORRECTED_PATH = RESULTS_DIR / "two_call_corrected_transcript.txt"
WORD_LIST_PATH = RESULTS_DIR / "two_call_word_list.txt"
DIFF_PATH = RESULTS_DIR / "two_call_diff.txt"
SCORE_PATH = RESULTS_DIR / "two_call_score.txt"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DETECTION_MODEL = "google/gemini-3.1-flash-lite"
CORRECTION_MODEL = "google/gemini-3.1-flash-lite"
MAX_CHUNK_WORDS = 400
MAX_HTTP_RETRIES = 3
MAX_GUARD_RETRIES = 1
MIN_SOURCE_SIMILARITY_PCT = 92.0
MAX_CHANGE_VOLUME_PCT = 8.0
FALLBACK_REJECTED_CORRECTIONS_TO_SOURCE = True
FALLBACK_REJECTED_DETECTIONS_TO_EMPTY = True
LAST_FALLBACK_CHUNKS: list[dict[str, object]] = []


DETECTION_SYSTEM_PROMPT = """You are an expert Islamic scholar and Arabic language specialist with deep knowledge of classical Arabic transliteration, Quranic phrases, Islamic terminology, and the works of Shaykh Abdul Hakeem Murad (Timothy Winter) of Cambridge Central Mosque.

Your task is to read the following ASR-generated transcript and identify all words or phrases that appear to be incorrectly transcribed. Focus especially on Arabic and Islamic terms, Quranic phrases, proper nouns, and honorifics. For each error found, return a structured word list in this exact format:
INCORRECT: [wrong word/phrase] | POSITION: [word index] | CORRECT: [correct form] | REASON: [brief reason]

CRITICAL ERROR-DEFINITION RULE: Do NOT flag words merely to add diacritical marks or make academic transliteration improvements. Only flag a word or phrase when it is clearly misheard or misspelled—meaning it sounds wrong in context or is a completely different word from what the speaker intended. For example, "Tasbih" is correct and must not be flagged, but "dhillowat" is wrong because it should be "tilawah"; "Bahloul" is wrong because it should be "Bahlul"; and "press" is wrong because it should be "prayer".

Positions are global, one-based word indexes from the supplied WORD POSITION GUIDE. For a multi-word phrase, give the inclusive range as [start-end]. Do not propose stylistic rewrites or corrections to ordinary English. If there are no errors, return exactly NO ERRORS. Return only the word list, nothing else."""


CORRECTION_SYSTEM_PROMPT = """You are a meticulous transcript editor with expertise in Islamic and Arabic terminology and the speaking style of Shaykh Abdul Hakeem Murad of Cambridge Central Mosque. Preserve every unrelated word verbatim. Do not paraphrase, summarize, or rewrite. Do not add or remove content. Return only the complete corrected transcript."""


DETECTION_USER_TEMPLATE = """TRANSCRIPT CHUNK:
<transcript start_word="[START_WORD]" end_word="[END_WORD]">
[FULL ASSEMBLYAI TRANSCRIPT CHUNK]
</transcript>

WORD POSITION GUIDE:
[START_WORD] [word at that global one-based position]
[START_WORD + 1] [next word]
...
[END_WORD] [last word in this chunk]

Identify every incorrectly transcribed Islamic/Arabic term, Quranic phrase, proper noun, or honorific in this transcript chunk. Use the global indexes in the WORD POSITION GUIDE. Return only entries in the exact format required by the system prompt, or NO ERRORS."""


CORRECTION_USER_TEMPLATE = """TRANSCRIPT CHUNK:
<transcript start_word="[START_WORD]" end_word="[END_WORD]">
[FULL ASSEMBLYAI TRANSCRIPT CHUNK]
</transcript>

WORD LIST FOR THIS CHUNK:
[ONLY CALL 1 ENTRIES WHOSE POSITIONS FALL IN THIS CHUNK, OR NO LISTED ERRORS]

INSTRUCTION:
Apply corrections from the word list at the stated positions where supported by context. Fix only listed or equally clear Islamic/Arabic ASR errors. Return only the complete corrected transcript chunk, with all unrelated wording preserved verbatim."""


DETECTION_RETRY_SUFFIX_TEMPLATE = """VALIDATION RETRY:
Your previous response was rejected because [VALIDATION ERROR]. Re-read the original transcript chunk and position guide above. Return only valid word-list lines in the exact required format, or NO ERRORS."""


CORRECTION_RETRY_SUFFIX_TEMPLATE = """VALIDATION RETRY:
Your previous result was rejected because [GUARDRAIL FAILURE]. Start again from the original transcript chunk above and make fewer, strictly targeted corrections."""


class PipelineError(RuntimeError):
    """Raised when an API response or validation check cannot be accepted."""


@dataclass(frozen=True)
class TranscriptChunk:
    number: int
    start_word: int
    end_word: int
    text: str
    separator_after: str


@dataclass(frozen=True)
class WordListEntry:
    incorrect: str
    start_word: int
    end_word: int
    correct: str
    reason: str

    def line(self) -> str:
        position = (
            str(self.start_word)
            if self.start_word == self.end_word
            else f"{self.start_word}-{self.end_word}"
        )
        return (
            f"INCORRECT: [{self.incorrect}] | POSITION: [{position}] | "
            f"CORRECT: [{self.correct}] | REASON: [{self.reason}]"
        )


@dataclass(frozen=True)
class ChangeAudit:
    source_words: int
    candidate_words: int
    changed_source_words: int
    changed_candidate_words: int
    change_volume_pct: float
    source_similarity_pct: float

    @property
    def accepted(self) -> bool:
        return (
            self.change_volume_pct < MAX_CHANGE_VOLUME_PCT
            and self.source_similarity_pct > MIN_SOURCE_SIMILARITY_PCT
        )

    def rejection_reason(self) -> str:
        problems: list[str] = []
        if self.change_volume_pct >= MAX_CHANGE_VOLUME_PCT:
            problems.append(
                f"change volume {self.change_volume_pct:.2f}% is not under "
                f"{MAX_CHANGE_VOLUME_PCT:.2f}%"
            )
        if self.source_similarity_pct <= MIN_SOURCE_SIMILARITY_PCT:
            problems.append(
                f"source similarity {self.source_similarity_pct:.2f}% is not above "
                f"{MIN_SOURCE_SIMILARITY_PCT:.2f}%"
            )
        return "; ".join(problems) or "accepted"


@dataclass(frozen=True)
class Score:
    word_count: int
    vocabulary_overlap_pct: float
    sequence_agreement_pct: float
    common_unique_words: int
    only_in_reference: int
    only_in_candidate: int


@dataclass(frozen=True)
class ApiUsage:
    stage: str
    requested_model: str
    actual_model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_credits: float | None
    generation_id: str


def read_required(path: Path) -> str:
    if not path.is_file():
        raise PipelineError(f"Required file not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise PipelineError(f"Required file is empty: {path}")
    return text


def transcribe_audio_with_assemblyai(audio_path: Path, output_path: Path) -> None:
    """Create the AssemblyAI source transcript used by the two LLM calls."""

    try:
        import assemblyai as aai
    except ImportError as exc:
        raise PipelineError(
            "The assemblyai package is required for --audio mode."
        ) from exc

    if not audio_path.is_file():
        raise PipelineError(f"Audio file not found: {audio_path}")
    aai.settings.api_key = require_env("ASSEMBLYAI_API_KEY", env_path=ENV_PATH)
    config = aai.TranscriptionConfig(
        speech_models=["universal-2"],
        punctuate=True,
        format_text=True,
    )
    print(f"AssemblyAI: transcribing {audio_path} with universal-2...", flush=True)
    started = time.time()
    transcript = aai.Transcriber(config=config).transcribe(str(audio_path))
    if transcript.status == aai.TranscriptStatus.error:
        raise PipelineError(f"AssemblyAI transcription failed: {transcript.error}")
    text = (transcript.text or "").strip()
    if not text:
        raise PipelineError("AssemblyAI returned an empty transcript.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(
        f"AssemblyAI completed in {time.time() - started:.1f}s: "
        f"{len(text.split())} words -> {output_path}",
        flush=True,
    )


def configure_run_paths(args: argparse.Namespace) -> Path | None:
    """Apply optional CLI source/output/reference paths to module-level reports."""

    global RESULTS_DIR, SOURCE_PATH, REFERENCE_PATH
    global CORRECTED_PATH, WORD_LIST_PATH, DIFF_PATH, SCORE_PATH

    audio_path = args.audio.resolve() if args.audio else None
    explicit_source = args.source.resolve() if args.source else None
    if args.output_dir:
        results_dir = args.output_dir.resolve()
    elif audio_path:
        results_dir = audio_path.parent
    elif explicit_source:
        results_dir = explicit_source.parent
    else:
        results_dir = VERSION_DIR

    RESULTS_DIR = results_dir
    SOURCE_PATH = (
        results_dir / "assemblyai_transcript.txt"
        if audio_path
        else explicit_source or PROJECT_DIR / "assemblyai_transcript.txt"
    )
    if args.no_reference:
        REFERENCE_PATH = None
    elif args.reference:
        REFERENCE_PATH = args.reference.resolve()

    CORRECTED_PATH = results_dir / "two_call_corrected_transcript.txt"
    WORD_LIST_PATH = results_dir / "two_call_word_list.txt"
    DIFF_PATH = results_dir / "two_call_diff.txt"
    SCORE_PATH = results_dir / "two_call_score.txt"
    return audio_path


def split_transcript(text: str) -> tuple[str, list[TranscriptChunk]]:
    """Split at word boundaries while preserving every inter-chunk character."""

    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return text, []
    prefix = text[: matches[0].start()]
    chunks: list[TranscriptChunk] = []
    for number, first_index in enumerate(
        range(0, len(matches), MAX_CHUNK_WORDS), start=1
    ):
        last_index = min(first_index + MAX_CHUNK_WORDS, len(matches)) - 1
        next_index = last_index + 1
        core_start = matches[first_index].start()
        core_end = matches[last_index].end()
        separator_end = (
            matches[next_index].start() if next_index < len(matches) else len(text)
        )
        chunks.append(
            TranscriptChunk(
                number=number,
                start_word=first_index + 1,
                end_word=last_index + 1,
                text=text[core_start:core_end],
                separator_after=text[core_end:separator_end],
            )
        )
    return prefix, chunks


def position_guide(chunk: TranscriptChunk) -> str:
    return "\n".join(
        f"{position} {word}"
        for position, word in enumerate(
            re.findall(r"\S+", chunk.text), start=chunk.start_word
        )
    )


def build_detection_user_message(
    chunk: TranscriptChunk, retry_feedback: str | None = None
) -> str:
    message = f"""TRANSCRIPT CHUNK:
<transcript start_word="{chunk.start_word}" end_word="{chunk.end_word}">
{html.escape(chunk.text, quote=False)}
</transcript>

WORD POSITION GUIDE:
{position_guide(chunk)}

Identify every incorrectly transcribed Islamic/Arabic term, Quranic phrase, proper noun, or honorific in this transcript chunk. Use the global indexes in the WORD POSITION GUIDE. Return only entries in the exact format required by the system prompt, or NO ERRORS."""
    if retry_feedback:
        message += f"""

VALIDATION RETRY:
Your previous response was rejected because {retry_feedback}. Re-read the original transcript chunk and position guide above. Return only valid word-list lines in the exact required format, or NO ERRORS."""
    return message


def build_correction_user_message(
    chunk: TranscriptChunk,
    entries: list[WordListEntry],
    retry_feedback: str | None = None,
) -> str:
    word_list = "\n".join(entry.line() for entry in entries) or "NO LISTED ERRORS"
    message = f"""TRANSCRIPT CHUNK:
<transcript start_word="{chunk.start_word}" end_word="{chunk.end_word}">
{html.escape(chunk.text, quote=False)}
</transcript>

WORD LIST FOR THIS CHUNK:
{word_list}

INSTRUCTION:
Apply corrections from the word list at the stated positions where supported by context. Fix only listed or equally clear Islamic/Arabic ASR errors. Return only the complete corrected transcript chunk, with all unrelated wording preserved verbatim."""
    if retry_feedback:
        message += f"""

VALIDATION RETRY:
Your previous result was rejected because {retry_feedback}. Start again from the original transcript chunk above and make fewer, strictly targeted corrections."""
    return message


def _response_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def clean_model_output(content: str, *, correction: bool) -> str:
    value = content.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3:
            value = "\n".join(lines[1:-1]).strip()
    if correction:
        match = re.fullmatch(
            r"<transcript(?:\s+[^>]*)?>(.*)</transcript>",
            value,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            value = match.group(1).strip()
        if value.lower().startswith("corrected transcript:"):
            value = value[len("corrected transcript:") :].lstrip()
    if not value:
        raise PipelineError("OpenRouter returned empty content.")
    return html.unescape(value) if correction else value


def call_openrouter(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    stage: str,
    usage_sink: list[ApiUsage],
    timeout_seconds: int = 180,
) -> tuple[str, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    last_error: Exception | None = None
    for attempt in range(1, MAX_HTTP_RETRIES + 1):
        request = urllib.request.Request(
            OPENROUTER_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "SMART LABS Two-Call Transcription Pipeline",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
            if data.get("error"):
                raise PipelineError(f"OpenRouter error: {data['error']}")
            choices = data.get("choices") or []
            if not choices:
                raise PipelineError("OpenRouter returned no completion choices.")
            choice = choices[0]
            actual_model = str(data.get("model") or model)
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            raw_cost = usage.get("cost")
            try:
                cost_credits = float(raw_cost) if raw_cost is not None else None
            except (TypeError, ValueError):
                cost_credits = None
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            usage_sink.append(
                ApiUsage(
                    stage=stage,
                    requested_model=model,
                    actual_model=actual_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=int(
                        usage.get("total_tokens")
                        or prompt_tokens + completion_tokens
                    ),
                    cost_credits=cost_credits,
                    generation_id=str(data.get("id") or "unknown"),
                )
            )
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                raise PipelineError(
                    f"Completion was truncated (model={actual_model})."
                )
            if finish_reason not in {None, "stop"}:
                raise PipelineError(
                    f"Completion ended with finish_reason={finish_reason!r} "
                    f"(model={actual_model})."
                )
            content = _response_content(choice.get("message", {}).get("content"))
            return content, actual_model
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = PipelineError(f"OpenRouter HTTP {exc.code}: {body[:1000]}")
            if exc.code != 429 and not 500 <= exc.code < 600:
                raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = PipelineError(f"Could not reach OpenRouter: {exc.reason}")
        except TimeoutError:
            last_error = PipelineError("The OpenRouter request timed out.")
        if attempt < MAX_HTTP_RETRIES:
            print(
                f"  API attempt {attempt}/{MAX_HTTP_RETRIES} failed; retrying...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(2 ** (attempt - 1))
    raise last_error or PipelineError("OpenRouter request failed.")


def parse_position(value: str, incorrect: str) -> tuple[int, int]:
    cleaned = value.strip().replace("–", "-").replace("—", "-")
    match = re.fullmatch(r"(\d+)(?:\s*(?:-|to)\s*(\d+))?", cleaned, re.I)
    if not match:
        raise PipelineError(f"Invalid word-list position: [{value}]")
    start = int(match.group(1))
    if match.group(2):
        end = int(match.group(2))
    else:
        end = start + max(len(re.findall(r"\S+", incorrect)), 1) - 1
    if end < start:
        raise PipelineError(f"Reversed word-list position: [{value}]")
    return start, end


def _normalized_phrase(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def parse_entry_line(line: str) -> tuple[str, str, str, str]:
    """Accept canonical bracketed fields and equivalent bare field values."""

    parts = re.split(r"\s*\|\s*", line.strip())
    labels = ("INCORRECT", "POSITION", "CORRECT", "REASON")
    if len(parts) != len(labels):
        raise PipelineError("expected exactly four pipe-separated fields")
    values: list[str] = []
    for part, label in zip(parts, labels, strict=True):
        match = re.fullmatch(rf"{label}\s*:\s*(.*)", part, re.I)
        if not match:
            raise PipelineError(f"expected {label}: field")
        value = match.group(1).strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1].strip()
        elif value.startswith("[") or value.endswith("]"):
            raise PipelineError(f"unbalanced brackets in {label} field")
        values.append(value)
    return values[0], values[1], values[2], values[3]


def parse_detection_output(
    content: str, chunk: TranscriptChunk
) -> list[WordListEntry]:
    cleaned = clean_model_output(content, correction=False)
    if cleaned.strip().upper() == "NO ERRORS":
        return []
    source_words = re.findall(r"\S+", chunk.text)
    entries: list[WordListEntry] = []
    for line_number, line in enumerate(cleaned.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            incorrect, position, correct, reason = parse_entry_line(line)
        except PipelineError as exc:
            raise PipelineError(
                f"Malformed detector line {line_number} ({exc}): {line[:160]!r}"
            ) from exc
        if not correct and "previous correction" in reason.casefold():
            continue
        if not all((incorrect, correct, reason)):
            raise PipelineError(f"Empty field in detector line {line_number}.")
        start, end = parse_position(position, incorrect)
        if start < chunk.start_word or end > chunk.end_word:
            raise PipelineError(
                f"Detector position {start}-{end} falls outside chunk "
                f"{chunk.start_word}-{chunk.end_word}."
            )
        local_start = start - chunk.start_word
        local_end = end - chunk.start_word + 1
        cited_source = " ".join(source_words[local_start:local_end])
        if _normalized_phrase(cited_source) != _normalized_phrase(incorrect):
            phrase_length = max(len(re.findall(r"\S+", incorrect)), 1)
            target = _normalized_phrase(incorrect)
            matching_offsets = [
                offset
                for offset in range(0, len(source_words) - phrase_length + 1)
                if _normalized_phrase(
                    " ".join(source_words[offset : offset + phrase_length])
                )
                == target
            ]
            if len(matching_offsets) != 1:
                raise PipelineError(
                    f"Detector phrase [{incorrect}] does not match source words "
                    f"{start}-{end}: [{cited_source}], and has "
                    f"{len(matching_offsets)} exact normalized matches in the chunk"
                )
            start = chunk.start_word + matching_offsets[0]
            end = start + phrase_length - 1
        entries.append(
            WordListEntry(incorrect, start, end, correct, reason)
        )
    if not entries:
        raise PipelineError("Detector returned neither word-list entries nor NO ERRORS.")
    return entries


def deduplicate_entries(entries: list[WordListEntry]) -> list[WordListEntry]:
    seen: set[tuple[int, int, str, str]] = set()
    result: list[WordListEntry] = []
    for entry in sorted(entries, key=lambda item: (item.start_word, item.end_word)):
        key = (
            entry.start_word,
            entry.end_word,
            entry.incorrect.casefold(),
            entry.correct.casefold(),
        )
        if key not in seen:
            seen.add(key)
            result.append(entry)
    return result


def transliteration_base(text: str) -> str:
    """Reduce diacritic/punctuation variants while preserving every script."""

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return "".join(character for character in without_marks if character.isalnum())


def substantive_entries(entries: list[WordListEntry]) -> list[WordListEntry]:
    """Drop no-ops and changes that only alter transliteration typography."""

    return [
        entry
        for entry in entries
        if transliteration_base(entry.incorrect)
        != transliteration_base(entry.correct)
    ]


def audit_change(source: str, candidate: str) -> ChangeAudit:
    source_words = re.findall(r"\S+", source)
    candidate_words = re.findall(r"\S+", candidate)
    matcher = difflib.SequenceMatcher(
        None, source_words, candidate_words, autojunk=False
    )
    changed_source = 0
    changed_candidate = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed_source += i2 - i1
            changed_candidate += j2 - j1
    source_pct = 100.0 * changed_source / max(len(source_words), 1)
    candidate_pct = 100.0 * changed_candidate / max(len(candidate_words), 1)
    return ChangeAudit(
        source_words=len(source_words),
        candidate_words=len(candidate_words),
        changed_source_words=changed_source,
        changed_candidate_words=changed_candidate,
        change_volume_pct=max(source_pct, candidate_pct),
        source_similarity_pct=100.0 * matcher.ratio(),
    )


def entries_for_chunk(
    entries: list[WordListEntry], chunk: TranscriptChunk
) -> list[WordListEntry]:
    return [
        entry
        for entry in entries
        if entry.start_word <= chunk.end_word and entry.end_word >= chunk.start_word
    ]


def retain_positioned_changes(
    source: str,
    candidate: str,
    chunk: TranscriptChunk,
    entries: list[WordListEntry],
) -> tuple[str, int]:
    """Restore model edits that are not localized to detected source positions."""

    source_matches = list(re.finditer(r"\S+", source))
    source_words = [match.group(0) for match in source_matches]
    candidate_words = re.findall(r"\S+", candidate)
    allowed_ranges = [
        (
            max(entry.start_word - chunk.start_word - 1, 0),
            min(entry.end_word - chunk.start_word + 2, len(source_words)),
        )
        for entry in entries
    ]
    matcher = difflib.SequenceMatcher(
        None, source_words, candidate_words, autojunk=False
    )
    patches: list[tuple[int, int, str]] = []
    discarded = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            allowed = any(start <= i1 <= end for start, end in allowed_ranges)
        else:
            allowed = any(
                i1 >= start and i2 <= end for start, end in allowed_ranges
            )
        if not allowed:
            discarded += 1
            continue
        if i1 < len(source_matches):
            char_start = source_matches[i1].start()
        else:
            char_start = len(source)
        char_end = (
            source_matches[i2 - 1].end() if i2 > i1 else char_start
        )
        replacement = " ".join(candidate_words[j1:j2])
        if tag == "insert" and replacement:
            if char_start == len(source):
                replacement = " " + replacement
            else:
                replacement += " "
        patches.append((char_start, char_end, replacement))

    sanitized = source
    for char_start, char_end, replacement in reversed(patches):
        sanitized = sanitized[:char_start] + replacement + sanitized[char_end:]
    return sanitized, discarded


def detect_errors(
    api_key: str,
    chunks: list[TranscriptChunk],
    usage_records: list[ApiUsage],
) -> tuple[list[WordListEntry], set[str], list[dict[str, object]]]:
    all_entries: list[WordListEntry] = []
    actual_models: set[str] = set()
    fallback_chunks: list[dict[str, object]] = []
    print(f"Call 1: detecting errors in {len(chunks)} chunks...", flush=True)
    for chunk in chunks:
        retry_feedback: str | None = None
        for guard_attempt in range(MAX_GUARD_RETRIES + 1):
            print(
                f"  Detection chunk {chunk.number}/{len(chunks)}"
                + (" (validation retry)" if guard_attempt else ""),
                flush=True,
            )
            try:
                content, actual_model = call_openrouter(
                    api_key=api_key,
                    model=DETECTION_MODEL,
                    system_prompt=DETECTION_SYSTEM_PROMPT,
                    user_message=build_detection_user_message(
                        chunk, retry_feedback
                    ),
                    max_tokens=3000,
                    stage="Call 1 - detection",
                    usage_sink=usage_records,
                )
                actual_models.add(actual_model)
                all_entries.extend(parse_detection_output(content, chunk))
                break
            except PipelineError as exc:
                if guard_attempt >= MAX_GUARD_RETRIES:
                    if FALLBACK_REJECTED_DETECTIONS_TO_EMPTY:
                        fallback_chunks.append(
                            {
                                "stage": "detection",
                                "chunk_number": chunk.number,
                                "start_word": chunk.start_word,
                                "end_word": chunk.end_word,
                                "rejection_reason": str(exc),
                                "action": "detection_entries_omitted_source_retained",
                                "manual_review_required": True,
                            }
                        )
                        print(
                            f"    Safety fallback: detection chunk {chunk.number} "
                            "failed validation twice; omitted its detection entries, "
                            "retained the source text, and flagged it for manual review.",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    raise PipelineError(
                        f"Detection chunk {chunk.number} was rejected twice: {exc}"
                    ) from exc
                retry_feedback = str(exc)
                print(f"    Rejected: {exc}", file=sys.stderr, flush=True)
    return deduplicate_entries(all_entries), actual_models, fallback_chunks


def correct_chunks(
    api_key: str,
    chunks: list[TranscriptChunk],
    entries: list[WordListEntry],
    usage_records: list[ApiUsage],
) -> tuple[list[str], list[ChangeAudit], set[str], int, list[dict[str, object]]]:
    corrected: list[str] = []
    audits: list[ChangeAudit] = []
    actual_models: set[str] = set()
    discarded_change_blocks = 0
    fallback_chunks: list[dict[str, object]] = []
    print(f"Call 2: correcting {len(chunks)} chunks...", flush=True)
    for chunk in chunks:
        relevant = entries_for_chunk(entries, chunk)
        retry_feedback: str | None = None
        for guard_attempt in range(MAX_GUARD_RETRIES + 1):
            rejected_audit: ChangeAudit | None = None
            print(
                f"  Correction chunk {chunk.number}/{len(chunks)} "
                f"({len(relevant)} detected entries)"
                + (" (guardrail retry)" if guard_attempt else ""),
                flush=True,
            )
            try:
                content, actual_model = call_openrouter(
                    api_key=api_key,
                    model=CORRECTION_MODEL,
                    system_prompt=CORRECTION_SYSTEM_PROMPT,
                    user_message=build_correction_user_message(
                        chunk, relevant, retry_feedback
                    ),
                    max_tokens=2500,
                    stage="Call 2 - correction",
                    usage_sink=usage_records,
                )
                actual_models.add(actual_model)
                candidate = clean_model_output(content, correction=True)
                candidate, discarded = retain_positioned_changes(
                    chunk.text, candidate, chunk, relevant
                )
                audit = audit_change(chunk.text, candidate)
                if not audit.accepted:
                    rejected_audit = audit
                    raise PipelineError(audit.rejection_reason())
                corrected.append(candidate)
                audits.append(audit)
                discarded_change_blocks += discarded
                break
            except PipelineError as exc:
                retry_feedback = str(exc)
                if guard_attempt >= MAX_GUARD_RETRIES:
                    if (
                        FALLBACK_REJECTED_CORRECTIONS_TO_SOURCE
                        and rejected_audit is not None
                    ):
                        fallback_chunks.append(
                            {
                                "chunk_number": chunk.number,
                                "start_word": chunk.start_word,
                                "end_word": chunk.end_word,
                                "rejection_reason": retry_feedback,
                                "rejected_change_volume_pct": round(
                                    rejected_audit.change_volume_pct, 6
                                ),
                                "rejected_source_similarity_pct": round(
                                    rejected_audit.source_similarity_pct, 6
                                ),
                                "action": "source_chunk_retained",
                                "manual_review_required": True,
                            }
                        )
                        corrected.append(chunk.text)
                        audits.append(audit_change(chunk.text, chunk.text))
                        print(
                            f"    Safety fallback: correction chunk {chunk.number} "
                            "failed preservation checks twice; retained the "
                            "unchanged source chunk and flagged it for manual review.",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    raise PipelineError(
                        f"Correction chunk {chunk.number} was rejected twice: "
                        f"{retry_feedback}"
                    ) from exc
                print(
                    f"    Rejected: {retry_feedback}",
                    file=sys.stderr,
                    flush=True,
                )
    return (
        corrected,
        audits,
        actual_models,
        discarded_change_blocks,
        fallback_chunks,
    )


def normalize_words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def score_transcript(reference: str, candidate: str) -> Score:
    reference_words = normalize_words(reference)
    candidate_words = normalize_words(candidate)
    reference_set = set(reference_words)
    candidate_set = set(candidate_words)
    common = reference_set & candidate_set
    union = reference_set | candidate_set
    return Score(
        word_count=len(candidate_words),
        vocabulary_overlap_pct=100.0 * len(common) / max(len(union), 1),
        sequence_agreement_pct=100.0
        * difflib.SequenceMatcher(
            None, reference_words, candidate_words, autojunk=False
        ).ratio(),
        common_unique_words=len(common),
        only_in_reference=len(reference_set - candidate_set),
        only_in_candidate=len(candidate_set - reference_set),
    )


def build_diff(source: str, corrected: str, audit: ChangeAudit) -> str:
    source_words = re.findall(r"\S+", source)
    corrected_words = re.findall(r"\S+", corrected)
    matcher = difflib.SequenceMatcher(
        None, source_words, corrected_words, autojunk=False
    )
    lines = [
        "TWO-CALL PIPELINE WORD-LEVEL DIFF",
        "=================================",
        f"Original: {SOURCE_PATH}",
        f"Corrected: {CORRECTED_PATH}",
        f"Source words: {audit.source_words}",
        f"Corrected words: {audit.candidate_words}",
        f"Change volume: {audit.change_volume_pct:.2f}% (must be < {MAX_CHANGE_VOLUME_PCT:.2f}%)",
        f"Source similarity: {audit.source_similarity_pct:.2f}% (must be > {MIN_SOURCE_SIMILARITY_PCT:.2f}%)",
        "",
        "Legend: - original words; + corrected words",
        "",
    ]
    change_number = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        change_number += 1
        context_before = " ".join(source_words[max(0, i1 - 8) : i1])
        context_after = " ".join(source_words[i2 : min(len(source_words), i2 + 8)])
        lines.extend(
            [
                f"Change {change_number} (original position {i1 + 1}):",
                f"  Context before: {context_before or '[start of transcript]'}",
                f"- {' '.join(source_words[i1:i2]) or '[nothing]'}",
                f"+ {' '.join(corrected_words[j1:j2]) or '[nothing]'}",
                f"  Context after: {context_after or '[end of transcript]'}",
                "",
            ]
        )
    if change_number == 0:
        lines.append("No word-level changes.")
    else:
        lines.insert(9, f"Total change blocks: {change_number}")
    return "\n".join(lines).rstrip() + "\n"


def build_score_report(
    source: str,
    corrected: str,
    audit: ChangeAudit,
    detection_models: set[str],
    correction_models: set[str],
    entry_count: int,
    applied_entry_count: int,
    discarded_change_blocks: int,
    fallback_chunks: list[dict[str, object]],
    usage_records: list[ApiUsage],
) -> str:
    detection_usage = [
        record for record in usage_records if record.stage == "Call 1 - detection"
    ]
    correction_usage = [
        record for record in usage_records if record.stage == "Call 2 - correction"
    ]

    def usage_summary(records: list[ApiUsage]) -> tuple[int, int, int, float, int]:
        return (
            sum(record.prompt_tokens for record in records),
            sum(record.completion_tokens for record in records),
            sum(record.total_tokens for record in records),
            sum(record.cost_credits or 0.0 for record in records),
            sum(record.cost_credits is None for record in records),
        )

    d_prompt, d_completion, d_total, d_cost, d_missing = usage_summary(
        detection_usage
    )
    c_prompt, c_completion, c_total, c_cost, c_missing = usage_summary(
        correction_usage
    )
    combined_cost_note = (
        ""
        if d_missing + c_missing == 0
        else f" ({d_missing + c_missing} response(s) did not report cost)"
    )
    lines = [
        "TWO-CALL PIPELINE EVALUATION",
        "============================",
        f"Detection model requested: {DETECTION_MODEL}",
        "Detection model(s) returned: " + ", ".join(sorted(detection_models)),
        f"Correction model requested: {CORRECTION_MODEL}",
        "Correction model(s) returned: " + ", ".join(sorted(correction_models)),
        f"Detected word-list entries: {entry_count}",
        f"Substantive entries supplied to Call 2: {applied_entry_count}",
        f"No-op/transliteration-style entries filtered: {entry_count - applied_entry_count}",
        f"Unpositioned model change blocks restored from source: {discarded_change_blocks}",
        f"Rejected correction chunks restored from source: {len(fallback_chunks)}",
        f"Manual-review chunks: {', '.join(str(item['chunk_number']) for item in fallback_chunks) or 'none'}",
        "",
        "OpenRouter usage (includes any rejected/retried completions):",
        f"  Call 1 requests: {len(detection_usage)}",
        f"  Call 1 tokens: {d_prompt} prompt + {d_completion} completion = {d_total} total",
        f"  Call 1 cost: {d_cost:.10f} credits"
        + (f" ({d_missing} response(s) missing cost)" if d_missing else ""),
        f"  Call 2 requests: {len(correction_usage)}",
        f"  Call 2 tokens: {c_prompt} prompt + {c_completion} completion = {c_total} total",
        f"  Call 2 cost: {c_cost:.10f} credits"
        + (f" ({c_missing} response(s) missing cost)" if c_missing else ""),
        f"  Combined tokens: {d_prompt + c_prompt} prompt + {d_completion + c_completion} completion = {d_total + c_total} total",
        f"  Combined cost: {d_cost + c_cost:.10f} credits{combined_cost_note}",
        "",
        "Correction guardrails (AssemblyAI source vs corrected output):",
        f"  Change volume: {audit.change_volume_pct:.2f}% (PASS: < {MAX_CHANGE_VOLUME_PCT:.2f}%)",
        f"  Source similarity: {audit.source_similarity_pct:.2f}% (PASS: > {MIN_SOURCE_SIMILARITY_PCT:.2f}%)",
        "",
    ]
    if REFERENCE_PATH is None:
        lines.extend(
            [
                "Reference-based evaluation disabled for this run.",
                "Only AssemblyAI-source preservation scores are available.",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"
    if not REFERENCE_PATH.is_file():
        lines.extend(
            [
                f"Evaluation unavailable: reference file not found at {REFERENCE_PATH}",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(
        [
            "The YouTube transcript is used below for evaluation only. It was not sent",
            "to either model and was not read until correction was complete.",
            "",
        ]
    )
    reference = read_required(REFERENCE_PATH)
    original_score = score_transcript(reference, source)
    corrected_score = score_transcript(reference, corrected)
    lines.extend(
        [
            "Scores against YouTube reference:",
            "",
            "Transcript                 Words   Vocabulary overlap   Sequence agreement",
            "-------------------------  ------  -------------------  ------------------",
            f"AssemblyAI original        {original_score.word_count:6d}  {original_score.vocabulary_overlap_pct:18.2f}%  {original_score.sequence_agreement_pct:17.2f}%",
            f"Two-call corrected         {corrected_score.word_count:6d}  {corrected_score.vocabulary_overlap_pct:18.2f}%  {corrected_score.sequence_agreement_pct:17.2f}%",
            f"Delta                              {corrected_score.vocabulary_overlap_pct - original_score.vocabulary_overlap_pct:+18.2f} pp  {corrected_score.sequence_agreement_pct - original_score.sequence_agreement_pct:+17.2f} pp",
            "",
            "Scoring method: lowercase, strip punctuation, split into words; vocabulary",
            "overlap is unique-word Jaccard similarity and sequence agreement is",
            "difflib.SequenceMatcher ratio with autojunk=False.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def preview_prompts() -> None:
    print("CALL 1 — ERROR DETECTION")
    print(f"Model: {DETECTION_MODEL}\n")
    print("SYSTEM PROMPT")
    print("-------------")
    print(DETECTION_SYSTEM_PROMPT)
    print("\nUSER MESSAGE TEMPLATE")
    print("---------------------")
    print(DETECTION_USER_TEMPLATE)
    print("\nVALIDATION-RETRY SUFFIX (only after a rejected response)")
    print("---------------------------------------------------------")
    print(DETECTION_RETRY_SUFFIX_TEMPLATE)
    print("\n\nCALL 2 — CORRECTION")
    print(f"Model: {CORRECTION_MODEL}\n")
    print("SYSTEM PROMPT")
    print("-------------")
    print(CORRECTION_SYSTEM_PROMPT)
    print("\nUSER MESSAGE TEMPLATE")
    print("---------------------")
    print(CORRECTION_USER_TEMPLATE)
    print("\nGUARDRAIL-RETRY SUFFIX (only after a rejected response)")
    print("--------------------------------------------------------")
    print(CORRECTION_RETRY_SUFFIX_TEMPLATE)


def run_pipeline(audio_path: Path | None = None) -> None:
    global LAST_FALLBACK_CHUNKS
    LAST_FALLBACK_CHUNKS = []
    load_env_file(ENV_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if audio_path is not None:
        transcribe_audio_with_assemblyai(audio_path, SOURCE_PATH)

    source = read_required(SOURCE_PATH)
    prefix, chunks = split_transcript(source)
    if not chunks:
        raise PipelineError("The AssemblyAI transcript contains no words.")
    if any(len(re.findall(r"\S+", chunk.text)) > MAX_CHUNK_WORDS for chunk in chunks):
        raise PipelineError("Internal chunking error: a chunk exceeds 400 words.")

    api_key = require_env("OPENROUTER_API_KEY", env_path=ENV_PATH)

    usage_records: list[ApiUsage] = []
    (
        entries,
        detection_models,
        detection_fallback_chunks,
    ) = detect_errors(api_key, chunks, usage_records)
    WORD_LIST_PATH.write_text(
        ("\n".join(entry.line() for entry in entries) or "NO ERRORS") + "\n",
        encoding="utf-8",
    )
    correction_entries = substantive_entries(entries)
    print(
        f"Call 1 produced {len(entries)} entries; supplying "
        f"{len(correction_entries)} substantive entries to Call 2 "
        f"({len(entries) - len(correction_entries)} no-op/style-only entries filtered).",
        flush=True,
    )
    (
        corrected_chunks,
        chunk_audits,
        correction_models,
        discarded_change_blocks,
        correction_fallback_chunks,
    ) = correct_chunks(
        api_key, chunks, correction_entries, usage_records
    )
    fallback_chunks = detection_fallback_chunks + correction_fallback_chunks
    LAST_FALLBACK_CHUNKS = fallback_chunks
    corrected = prefix + "".join(
        candidate + chunk.separator_after
        for candidate, chunk in zip(corrected_chunks, chunks, strict=True)
    )
    full_audit = audit_change(source, corrected)
    if not full_audit.accepted:
        raise PipelineError(
            "Combined transcript failed guardrails after all chunks passed: "
            + full_audit.rejection_reason()
        )

    CORRECTED_PATH.write_text(corrected, encoding="utf-8")
    DIFF_PATH.write_text(build_diff(source, corrected, full_audit), encoding="utf-8")
    SCORE_PATH.write_text(
        build_score_report(
            source,
            corrected,
            full_audit,
            detection_models,
            correction_models,
            len(entries),
            len(correction_entries),
            discarded_change_blocks,
            fallback_chunks,
            usage_records,
        ),
        encoding="utf-8",
    )
    print("\nPipeline completed successfully.")
    print(f"Word list: {WORD_LIST_PATH}")
    print(f"Corrected transcript: {CORRECTED_PATH}")
    print(f"Diff: {DIFF_PATH}")
    print(f"Score: {SCORE_PATH}")
    print(
        f"Guardrails: change volume {full_audit.change_volume_pct:.2f}%; "
        f"source similarity {full_audit.source_similarity_pct:.2f}%"
    )
    print(f"Accepted correction chunks: {len(chunk_audits)}/{len(chunks)}")
    print(f"Chunks retained from source for manual review: {len(fallback_chunks)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview-prompts",
        action="store_true",
        help="print both system prompts and user-message templates without API calls",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--audio",
        type=Path,
        help="audio file to transcribe with AssemblyAI universal-2 before correction",
    )
    source_group.add_argument(
        "--source",
        type=Path,
        help="existing AssemblyAI transcript to correct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for the AssemblyAI transcript and two-call output files",
    )
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--reference",
        type=Path,
        help="optional reference transcript used only after correction for evaluation",
    )
    reference_group.add_argument(
        "--no-reference",
        action="store_true",
        help="disable reference-based evaluation and report preservation only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preview_prompts:
        preview_prompts()
        return 0
    try:
        audio_path = configure_run_paths(args)
        run_pipeline(audio_path)
    except (PipelineError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
