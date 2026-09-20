# V6 Database Schema

## Purpose

The database preserves the complete path from a source video to a corrected
transcript. It supports reproducible experiments without treating source
similarity as proof of transcription accuracy.

## Data relationships

1. `videos` stores one canonical record per source video.
2. `transcripts` stores every distinct transcript version and its checksum.
3. `transcript_chunks` stores ordered, overlapping sections of a transcript
   using zero-based, half-open word positions: `start_word` is included and
   `end_word` is excluded.
4. `pipeline_runs` stores the configuration and status of each import,
   transcription, correction, or evaluation run.
5. `model_calls` stores each numbered LLM call, its model, prompt version,
   request and response, token counts, cost, and status. API secrets are never
   stored.
6. `corpus_terms` stores canonical terminology.
7. `corpus_variants` separately labels accepted spellings, observed ASR
   errors, and generated possible errors.
8. `corpus_evidence` links a term or variant to a video, timestamp, context,
   and reviewer. A record cannot be marked verified without a reviewer and
   review time.
9. `correction_proposals` stores word-level changes proposed for a chunk and
   any supporting corpus term.
10. `correction_decisions` records whether each proposal was accepted,
    rejected, or modified.

## Chunk position convention

For a 400-word first chunk, the bounds are `start_word = 0` and
`end_word = 400`. With a 100-word overlap, the next chunk starts at word 300.
Its bounds are `start_word = 300` and `end_word = 700`. The stored
`word_count` must equal `end_word - start_word`.

Stage 3 will implement these calculations. Step 2 defines and validates only
the storage contract.

## Integrity protections

- Foreign keys prevent orphan transcripts, chunks, evidence, calls, and
  corrections.
- Controlled status and category values prevent inconsistent labels.
- Unique constraints prevent duplicate videos, transcript versions, chunk
  positions, variants, and decisions.
- Text and chunk checksums must be lowercase SHA-256 values.
- JSON configuration, requests, and responses must be valid JSON.
- Timestamp and word-boundary checks reject impossible ranges.
- Migration checksums prevent silent changes to an applied schema.

