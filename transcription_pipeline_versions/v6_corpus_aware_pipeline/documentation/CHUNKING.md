# Overlapping Transcript Chunks

## Fixed v6 rule

- Maximum chunk size: 400 words.
- Contextual overlap: 100 words.
- Forward movement: 300 words.
- Word positions: zero-based and half-open.

For example, the first two full chunks cover words `0:400` and `300:700`.
Words `300:400` appear in both API contexts but retain the same global word
positions.

## Why global positions matter

The overlap is context supplied to both calls; it is not additional transcript
content. Proposed corrections are mapped back to global word ranges. Identical
proposals from neighbouring chunks are deduplicated, while contradictory or
partially overlapping proposals stop processing for review. Accepted changes
are applied once to the original transcript.

## Stored values

Each chunk records:

- its transcript and sequential index;
- exact text;
- global start and end word positions;
- word count;
- overlap before and after the chunk;
- optional audio start and end milliseconds; and
- a SHA-256 checksum.

Internal whitespace is preserved in each chunk. Word timings are accepted when
the transcription provider supplies them; otherwise timestamp fields remain
empty.

## Safety checks

- Invalid sizes or overlaps are rejected.
- Word timing counts and chronology are validated.
- Every persisted chunk set must cover the transcript continuously.
- Text and stored transcript checksums must agree.
- Re-running the same chunking job is idempotent.
- Existing chunks are never silently overwritten by different content.

Stage 3 uses synthetic offline tests only. Importing and showing a real
transcript row belongs to Stage 4.

