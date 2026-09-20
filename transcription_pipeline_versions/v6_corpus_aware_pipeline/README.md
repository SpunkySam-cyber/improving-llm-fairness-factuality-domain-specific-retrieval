# v6 Corpus-Aware Transcription Pipeline

This workspace is the clean successor to the v5 batch-ingestion experiment.
It will add persistent transcript chunks, 100-word contextual overlap, a
manually verified Islamic terminology corpus, and reproducible model
comparison records.

## Current status

Stages 1 through 4 are complete: the isolated workspace, normalized SQLite
schema, overlapping-chunk implementation, and one offline real-transcript
database proof exist. One previously completed AssemblyAI pilot transcript and
its five chunks are stored. No corpus entries, new API calls, media downloads,
or new transcription have been performed in v6.

Stage 5 is in progress. The 50-term manual-review workbook, review guide, and
offline validator are ready; corpus records remain empty until manually
verified entries are supplied.

The existing v5 code and research artifacts remain unchanged at:

`../v5_batch_ingestion_pipeline/`

## Directory roles

- `src/`: application and pipeline source code only.
- `database/`: `pipeline_v6.sqlite`, numbered migrations, and database notes.
- `corpus/`: manually reviewed terminology input and validated exports.
- `tests/`: offline unit and integration tests.
- `reports/`: generated evaluation reports and database screenshots.
- `documentation/`: schema, methodology, decisions, and operating guides.

## Agreed constraints

- Preserve v5 as the baseline; do not overwrite it.
- Begin with 400-word chunks and 100-word contextual overlap.
- Store all chunks with global word positions and checksums.
- Do not concatenate overlapping text directly when rebuilding transcripts.
- Keep accepted spellings separate from observed ASR errors and generated
  possible error variants.
- Require video and timestamp evidence before a corpus term is verified.
- Pilot the corpus with 50 verified terms before scaling to 1,000 or more.
- Defer the third LLM call and RAG question generation.
- Require explicit approval before paid API calls or further lecture batches.

See `documentation/IMPLEMENTATION_PLAN.md` for the gated work sequence.
