# Stage 4 Offline Database Proof

## Imported lecture

- Video ID: `6TnPRgY_loc`
- Title: Hope Despite the Westernization of Islamic Education - Abdal Hakim Murad
- Source: <https://www.youtube.com/watch?v=6TnPRgY_loc>
- Duration: 663 seconds
- Transcript provider: AssemblyAI (`universal-2`)
- Transcript words: 1,491
- Word timings: 1,491

This lecture was selected because its completed AssemblyAI artifacts already
existed in the v5 pilot and it was among the lectures manually reviewed for
quality. The saved text exactly matched the text in the saved AssemblyAI JSON
response before import.

## Stored result

The transcript produced five chunks using the fixed v6 rule:

| Chunk | Global word range | Words | Overlap before | Overlap after |
|---:|---:|---:|---:|---:|
| 0 | `0:400` | 400 | 0 | 100 |
| 1 | `300:700` | 400 | 100 | 100 |
| 2 | `600:1000` | 400 | 100 | 100 |
| 3 | `900:1300` | 400 | 100 | 100 |
| 4 | `1200:1491` | 291 | 100 | 0 |

The colon notation is zero-based and half-open: `0:400` contains word
positions 0 through 399.

## Verification

- Database integrity check: passed.
- Foreign-key violations: zero.
- Continuous reconstruction: all 1,491 global words reproduced exactly.
- Transcript checksum: matched.
- All five chunk checksums: matched.
- All chunks contain start and end audio timestamps.
- Model calls: zero.
- API cost: USD 0.00.

## Generated evidence

- `reports/stage4_import_report.json`: machine-readable provenance and checks.
- `reports/chunk_database_view.json`: all 13 chunk columns and the complete
  first stored row.
- `reports/chunk_database_view.png`: presentation-ready image of the same row.

The import is idempotent. Re-running it verifies and reuses the existing video,
transcript, chunks, and import-run records instead of duplicating them.

