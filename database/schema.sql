CREATE TABLE videos (
    video_id TEXT PRIMARY KEY,
    canonical_url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    channel_name TEXT,
    channel_id TEXT,
    duration_seconds INTEGER CHECK (
        duration_seconds IS NULL OR duration_seconds >= 0
    ),
    upload_date TEXT,
    content_type TEXT NOT NULL DEFAULT 'unknown' CHECK (
        content_type IN (
            'lecture', 'q_and_a', 'interview', 'discussion', 'other', 'unknown'
        )
    ),
    source_collection TEXT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    updated_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
) STRICT;

CREATE TABLE transcripts (
    transcript_id INTEGER PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE RESTRICT,
    source_type TEXT NOT NULL CHECK (
        source_type IN (
            'assemblyai', 'youtube_caption', 'manual', 'imported', 'llm_corrected'
        )
    ),
    provider TEXT,
    provider_transcript_id TEXT,
    language_code TEXT NOT NULL DEFAULT 'en',
    transcript_text TEXT NOT NULL,
    word_count INTEGER NOT NULL CHECK (word_count >= 0),
    text_sha256 TEXT NOT NULL CHECK (
        length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    parent_transcript_id INTEGER REFERENCES transcripts(transcript_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    UNIQUE (video_id, source_type, text_sha256)
) STRICT;

CREATE TABLE transcript_chunks (
    chunk_id INTEGER PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(transcript_id) ON DELETE RESTRICT,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    chunk_text TEXT NOT NULL,
    start_word INTEGER NOT NULL CHECK (start_word >= 0),
    end_word INTEGER NOT NULL CHECK (end_word > start_word),
    word_count INTEGER NOT NULL CHECK (
        word_count > 0 AND word_count = end_word - start_word
    ),
    overlap_before_words INTEGER NOT NULL DEFAULT 0 CHECK (
        overlap_before_words >= 0 AND overlap_before_words < word_count
    ),
    overlap_after_words INTEGER NOT NULL DEFAULT 0 CHECK (
        overlap_after_words >= 0 AND overlap_after_words < word_count
    ),
    start_time_ms INTEGER CHECK (start_time_ms IS NULL OR start_time_ms >= 0),
    end_time_ms INTEGER CHECK (
        end_time_ms IS NULL OR end_time_ms >= 0
    ),
    chunk_sha256 TEXT NOT NULL CHECK (
        length(chunk_sha256) = 64 AND chunk_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    CHECK (
        start_time_ms IS NULL OR end_time_ms IS NULL OR end_time_ms >= start_time_ms
    ),
    UNIQUE (transcript_id, chunk_index),
    UNIQUE (transcript_id, start_word, end_word)
) STRICT;

CREATE TABLE pipeline_runs (
    run_id TEXT PRIMARY KEY,
    pipeline_version TEXT NOT NULL,
    run_type TEXT NOT NULL CHECK (
        run_type IN ('transcription', 'correction', 'evaluation', 'import')
    ),
    source_transcript_id INTEGER REFERENCES transcripts(transcript_id) ON DELETE RESTRICT,
    output_transcript_id INTEGER REFERENCES transcripts(transcript_id) ON DELETE RESTRICT,
    configuration_json TEXT NOT NULL CHECK (json_valid(configuration_json)),
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'cancelled')
    ),
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    CHECK (
        completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at
    )
) STRICT;

CREATE TABLE model_calls (
    model_call_id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE RESTRICT,
    chunk_id INTEGER REFERENCES transcript_chunks(chunk_id) ON DELETE RESTRICT,
    call_number INTEGER NOT NULL CHECK (call_number BETWEEN 1 AND 3),
    purpose TEXT NOT NULL CHECK (
        purpose IN (
            'identify_errors', 'apply_corrections', 'corpus_verification',
            'evaluation', 'other'
        )
    ),
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_text_sha256 TEXT NOT NULL CHECK (
        length(input_text_sha256) = 64
        AND input_text_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    request_json TEXT NOT NULL CHECK (json_valid(request_json)),
    response_json TEXT CHECK (response_json IS NULL OR json_valid(response_json)),
    provider_request_id TEXT,
    input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
    cost_usd REAL CHECK (cost_usd IS NULL OR cost_usd >= 0),
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'completed', 'failed', 'skipped')
    ),
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    CHECK (
        completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at
    ),
    UNIQUE (run_id, chunk_id, call_number, purpose)
) STRICT;

CREATE TABLE corpus_terms (
    term_id INTEGER PRIMARY KEY,
    canonical_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL CHECK (
        category IN (
            'islamic_term', 'quranic_phrase', 'person_name', 'book_title',
            'group_name', 'honorific', 'place_name', 'other'
        )
    ),
    language_code TEXT,
    definition TEXT,
    notes TEXT,
    status TEXT NOT NULL DEFAULT 'draft' CHECK (
        status IN ('draft', 'verified', 'retired')
    ),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    updated_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
) STRICT;

CREATE TABLE corpus_variants (
    variant_id INTEGER PRIMARY KEY,
    term_id INTEGER NOT NULL REFERENCES corpus_terms(term_id) ON DELETE RESTRICT,
    variant_text TEXT NOT NULL,
    normalized_variant TEXT NOT NULL,
    variant_type TEXT NOT NULL CHECK (
        variant_type IN (
            'accepted_spelling', 'observed_asr_error', 'generated_possible_error'
        )
    ),
    verification_status TEXT NOT NULL DEFAULT 'pending' CHECK (
        verification_status IN ('pending', 'verified', 'rejected')
    ),
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    UNIQUE (term_id, normalized_variant, variant_type),
    UNIQUE (variant_id, term_id)
) STRICT;

CREATE TABLE corpus_evidence (
    evidence_id INTEGER PRIMARY KEY,
    term_id INTEGER NOT NULL REFERENCES corpus_terms(term_id) ON DELETE RESTRICT,
    variant_id INTEGER,
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE RESTRICT,
    transcript_id INTEGER REFERENCES transcripts(transcript_id) ON DELETE RESTRICT,
    start_time_ms INTEGER NOT NULL CHECK (start_time_ms >= 0),
    end_time_ms INTEGER CHECK (
        end_time_ms IS NULL OR end_time_ms >= start_time_ms
    ),
    context_excerpt TEXT NOT NULL,
    evidence_kind TEXT NOT NULL CHECK (
        evidence_kind IN ('spoken_audio', 'on_screen_text', 'published_source')
    ),
    review_status TEXT NOT NULL DEFAULT 'pending' CHECK (
        review_status IN ('pending', 'verified', 'rejected')
    ),
    reviewed_by TEXT,
    reviewed_at TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    FOREIGN KEY (variant_id, term_id)
        REFERENCES corpus_variants(variant_id, term_id) ON DELETE RESTRICT,
    CHECK (
        (review_status = 'verified' AND reviewed_by IS NOT NULL AND reviewed_at IS NOT NULL)
        OR review_status <> 'verified'
    )
) STRICT;

CREATE TABLE correction_proposals (
    proposal_id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE RESTRICT,
    model_call_id INTEGER NOT NULL REFERENCES model_calls(model_call_id) ON DELETE RESTRICT,
    chunk_id INTEGER NOT NULL REFERENCES transcript_chunks(chunk_id) ON DELETE RESTRICT,
    start_word INTEGER NOT NULL CHECK (start_word >= 0),
    end_word INTEGER NOT NULL CHECK (end_word > start_word),
    original_text TEXT NOT NULL,
    proposed_text TEXT NOT NULL,
    reason TEXT,
    confidence REAL CHECK (confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0),
    corpus_term_id INTEGER REFERENCES corpus_terms(term_id) ON DELETE RESTRICT,
    corpus_variant_id INTEGER REFERENCES corpus_variants(variant_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    UNIQUE (model_call_id, start_word, end_word, proposed_text)
) STRICT;

CREATE TABLE correction_decisions (
    decision_id INTEGER PRIMARY KEY,
    proposal_id INTEGER NOT NULL UNIQUE
        REFERENCES correction_proposals(proposal_id) ON DELETE RESTRICT,
    decision TEXT NOT NULL CHECK (
        decision IN ('accepted', 'rejected', 'modified')
    ),
    final_text TEXT,
    decided_by TEXT NOT NULL,
    rationale TEXT,
    decided_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    CHECK (
        (decision = 'modified' AND final_text IS NOT NULL)
        OR decision <> 'modified'
    )
) STRICT;

CREATE INDEX idx_videos_content_type
    ON videos(content_type);
CREATE INDEX idx_transcripts_video
    ON transcripts(video_id, created_at);
CREATE INDEX idx_chunks_transcript_order
    ON transcript_chunks(transcript_id, chunk_index);
CREATE INDEX idx_chunks_global_words
    ON transcript_chunks(transcript_id, start_word, end_word);
CREATE INDEX idx_runs_source_transcript
    ON pipeline_runs(source_transcript_id, created_at);
CREATE INDEX idx_model_calls_run
    ON model_calls(run_id, call_number, status);
CREATE INDEX idx_corpus_terms_category_status
    ON corpus_terms(category, status);
CREATE INDEX idx_corpus_variants_lookup
    ON corpus_variants(normalized_variant, verification_status);
CREATE INDEX idx_corpus_evidence_term
    ON corpus_evidence(term_id, review_status);
CREATE INDEX idx_correction_proposals_run_chunk
    ON correction_proposals(run_id, chunk_id, start_word);

