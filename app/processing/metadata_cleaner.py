
from collections import defaultdict


def clean_metadata(records: list,MIN_SCORE,MIN_TEXT_LEN,GARBAGE_TOKENS) -> list:
    cleaned = []

    for record in records:
        text  = record.get("text", "").strip()
        meta  = record.get("metadata", {})
        score = float(meta.get("scores", 0))
        topics = meta.get("topics", [])

        # normalize topics → always a list
        if isinstance(topics, str):
            topics = [topics] if topics else []

        # filters
        if score < MIN_SCORE:                              continue
        if len(text) < MIN_TEXT_LEN:                       continue
        if any(g in text for g in GARBAGE_TOKENS):         continue

        cleaned.append({
            "text": text,
            "metadata": {
                "page":        meta.get("page"),
                "sentence_id": meta.get("sentence_id"),
                "language":    meta.get("language", "ne"),
                "source":      meta.get("source"),
                "topics":      topics,
                "scores":      round(score, 4),
                "fiscal_year": "2081",
            }
        })

    return cleaned