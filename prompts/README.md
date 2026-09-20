# Prompts

These are the six prompts used by the two-call correction engine
(`transcription_pipeline_versions/v4_assemblyai_two_call_pipeline/two_call_pipeline.py`),
copied out of the code so they can be read on their own. The code is the
source of truth; run `python two_call_pipeline.py --preview-prompts` to print
the live versions (no API call is made).

| File | Used in |
| --- | --- |
| `call1_detection_system.txt`, `call1_detection_user_template.txt` | Call 1: detect likely errors and list them with positions |
| `call1_retry_suffix.txt` | Appended when Call 1 output is malformed |
| `call2_correction_system.txt`, `call2_correction_user_template.txt` | Call 2: apply only the listed corrections |
| `call2_retry_suffix.txt` | Appended when a chunk fails the preservation checks |

Bracketed placeholders such as `[START_WORD]` are filled in by the code at run time.
