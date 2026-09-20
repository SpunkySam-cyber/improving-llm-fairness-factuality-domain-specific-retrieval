# Improving LLM Fairness and Factuality through Domain-Specific Retrieval

Code for the first layer of a domain-specific question-answering system on Islamic lectures: collecting a verified lecture dataset, transcribing audio, and correcting Islamic/Arabic transcription errors with a two-call LLM pipeline. **Read `docs/TECHNICAL_REPORT.md` first.** It covers all six versions (V1 to V6), the results, the limits, and the planned work.

**Status:** research code, paused and handed over. There is no chatbot, no fairness result and no proven accuracy figure here. Agreement with a YouTube reference transcript measures agreement, not accuracy.

## What is in this repository

| Folder | Purpose |
|---|---|
| `transcription_pipeline_versions/v4_assemblyai_two_call_pipeline/` | Final correction engine (`two_call_pipeline.py`) |
| `transcription_pipeline_versions/v5_batch_ingestion_pipeline/` | Audio download and check, one-lecture runner, 10-lecture batch runner, `sources.json`, tests |
| `transcription_pipeline_versions/v6_corpus_aware_pipeline/` | Overlapping chunks, SQLite schema, term-list checker, tests, notes (not yet connected to the engine) |
| `tools/dataset/` | Scripts to find, screen, verify and freeze the video list, plus tests |
| `tools/dialogue/` | Scripts to extract draft question-and-answer exchanges, plus the record schema |
| `prompts/` | The six prompts, copied from the code for reading |
| `examples/` | A small synthetic text and the Call 1 output format |
| `database/schema.sql` | The 10-table database design |
| `docs/` | Technical report and diagrams |

Earlier experiments (V1 to V3), automation scripts and PDF builders are not published. V1 to V3 are described in the report.

## Setup

Python 3.11+, FFmpeg/ffprobe, and (for paid runs) AssemblyAI and OpenRouter keys.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt     # Windows
cp .env.example .env                                         # then add your own keys; never commit .env
```

## Try it

Offline tests (no internet, no cost):

```
python -m unittest discover -s transcription_pipeline_versions/v5_batch_ingestion_pipeline/tests
python -m unittest discover -s transcription_pipeline_versions/v6_corpus_aware_pipeline/tests
python -m unittest discover -s tools/dataset/tests
```

Show the prompts without calling any service:

```
python transcription_pipeline_versions/v4_assemblyai_two_call_pipeline/two_call_pipeline.py --preview-prompts
```

Transcribe and correct one audio file (**paid**: AssemblyAI + OpenRouter; check the cost first):

```
python transcription_pipeline_versions/v4_assemblyai_two_call_pipeline/two_call_pipeline.py --audio path/to/lecture.mp3
```

## Known limits

- The batch scripts and `tools/` need the frozen video list and generated data folders, which are not in this repository.
- V6 is not connected to the correction engine. The term corpus (1,000+ terms) is planned, not built.
- `import_existing_pilot.py` reads local pilot output that is not included; its test uses synthetic data.
- Keys must be supplied by the user. No audio, transcripts, databases or keys are stored here.

## License

All rights reserved for now. See `LICENSE`.
