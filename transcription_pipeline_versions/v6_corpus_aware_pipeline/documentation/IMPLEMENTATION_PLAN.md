# V6 Implementation Plan

## Objective

Create a research-auditable transcription correction pipeline in which every
source transcript, overlapping chunk, model call, terminology match,
correction decision, and final output can be traced to its origin.

## Stage 1: Isolated workspace

Status: complete.

- Create the v6 directory structure.
- Document directory ownership and constraints.
- Leave v5 unchanged as the verified baseline.

## Stage 2: Database schema

Status: complete.

Create `database/pipeline_v6.sqlite` with normalized tables for:

- videos;
- transcripts;
- transcript chunks;
- pipeline runs;
- model calls;
- correction proposals and decisions;
- canonical corpus terms;
- accepted and incorrect term variants; and
- video and timestamp evidence.

Add schema migrations, constraints, indexes, and offline schema tests.

## Stage 3: Overlapping chunker

Status: complete.

- Use a maximum context window of 400 words.
- Share 100 words between adjacent windows.
- Advance by 300 words per chunk.
- Preserve global word positions and, when available, audio timestamps.
- Persist source text, word counts, boundaries, and SHA-256 checksums.
- Merge correction proposals by global position so overlap is never duplicated
  in the reconstructed transcript.

Based on the completed Batch 1 speaking-rate average, a one-hour lecture is
expected to contain roughly 8,057 words: about 21 current non-overlapping
chunks or about 27 chunks with the planned overlap.

## Stage 4: Offline proof and screenshot

Status: complete.

- Import one previously generated AssemblyAI transcript.
- Create and persist its chunks without making API calls.
- Verify coverage, overlap, ordering, and reconstruction.
- Produce the requested view of `chunks` column names and one real row.

## Stage 5: Terminology corpus pilot

Status: in progress — workbook and validator ready; awaiting manual review.

- Prepare separate Terms, Variants, and Evidence inputs.
- Record canonical terms, legitimate alternatives, observed ASR errors, and
  generated possible errors as different variant types.
- Require a video URL, timestamp, and context excerpt for verification.
- Validate a 50-term pilot before expanding to 1,000 or more unique terms,
  names, phrases, book titles, group names, and honorifics.

## Stage 6: Corpus integration experiment

Status: pending.

Compare on the same manually reviewed material:

1. Call 1, corpus verification or enrichment, then Call 2.
2. Call 1, relevant corpus retrieval, then corpus-grounded Call 2.

Do not add a third call until its purpose and evidence requirement are defined.

## Stage 7: Model comparison

Status: pending approval and budget review.

Compare Gemini 3.1 Flash Lite, Gemini 3.5 Flash, and Gemini 3.7 Flash on the
same fixed sample. Record error-detection precision, missed errors, correction
accuracy, unrelated changes, token usage, cost, duration, and guardrail
failures.

## Stage 8: Production decision

Status: pending.

- Select the corpus placement and model from measured results.
- Freeze prompts, schema, corpus version, and evaluation data.
- Update research documentation.
- Resume lecture batches only after explicit approval.

Future RAG question generation and voice recognition remain outside the current
pipeline milestone.
