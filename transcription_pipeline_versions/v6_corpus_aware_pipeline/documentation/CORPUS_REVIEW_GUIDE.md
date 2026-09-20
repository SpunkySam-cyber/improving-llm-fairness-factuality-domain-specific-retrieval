# Manual Corpus Review Guide

## Goal

Create a pilot corpus of 50 unique, manually verified terms before expanding
to 1,000 or more. Every verified term must be traceable to audio that you
personally checked.

## What counts as a corpus term

- Islamic technical terms and multiword expressions;
- Quranic phrases;
- person and scholar names;
- book titles;
- schools, movements, or group names;
- honorifics; and
- relevant place names.

Use one canonical row for each unique item. A legitimate alternative spelling
is a variant, not a second canonical term.

## Your workflow

1. Open `SmartLabs_Corpus_Pilot_50_Terms.xlsx`.
2. Watch a selected Sheikh Abdal Hakim Murad video.
3. When you hear a relevant item, confirm the correct form from the audio and
   reliable context.
4. Add it to `Terms` and leave its status as `draft` initially.
5. Add legitimate alternative spellings to `Variants` as
   `accepted_spelling`.
6. If the saved transcript has an actual error, add that exact wrong form as
   `observed_asr_error`. Do not guess what the ASR might get wrong.
7. If possible error forms are generated later, label them
   `generated_possible_error` and keep them `pending`.
8. Add at least one `Evidence` row with the video ID, title, URL, timestamp,
   short context excerpt, your name, and review date.
9. Mark the evidence `verified` only after replaying the audio.
10. Mark the term `verified` only after it has verified evidence.

## Timestamp format

Enter timestamps as seconds. Decimals are allowed. `75` means 1 minute 15
seconds; `75.5` means 1 minute 15.5 seconds. The future database importer will
convert these values to milliseconds.

## Important distinctions

- `accepted_spelling`: a legitimate way to write the correct term.
- `observed_asr_error`: an incorrect form actually found in a transcript.
- `generated_possible_error`: a predicted mistake not yet observed.

Generated possibilities are never automatically treated as truth. A verified
observed error requires evidence linked directly to that variant.

## Validation

You can fill the workbook gradually. The validator reports progress and points
to the exact sheet row containing a problem. Database import remains blocked
until there are 50 verified unique terms, no validation errors, and verified
evidence for every verified term.

