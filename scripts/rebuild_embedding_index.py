from __future__ import annotations

from src.config import AppConfig
from src.db import get_record, init_db, list_records, upsert_record_embedding
from src.local_ai import embed_texts, record_embedding_text, text_hash


def main() -> None:
    init_db()
    config = AppConfig.from_env()
    records = [get_record(record["id"]) for record in list_records(sort_by="recent_updated")]
    records = [record for record in records if record]
    texts = []
    hashes = []
    for record in records:
        memo = record.get("edited_memo") or record.get("memo") or {}
        text = record_embedding_text(record, memo)
        texts.append(text)
        hashes.append(text_hash(text))

    model, vectors = embed_texts(texts, config)
    for record, vector, source_hash in zip(records, vectors, hashes):
        upsert_record_embedding(
            record_id=record["id"],
            model=model,
            embedding=vector,
            source_hash=source_hash,
        )
    print(f"Indexed {len(vectors)} records with {model}.")


if __name__ == "__main__":
    main()
