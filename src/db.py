from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from src.config import DATA_DIR, DB_PATH
from src.models import ReflectionInput


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def open_connection() -> Iterator[sqlite3.Connection]:
    connection = get_connection()
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with open_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scrapbook_works (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT NOT NULL,
                content_type TEXT NOT NULL,
                creator TEXT,
                status TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scrapbook_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                work_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                content_type TEXT NOT NULL,
                title TEXT NOT NULL,
                creator TEXT,
                appreciation_date TEXT,
                status TEXT NOT NULL,
                work_context TEXT DEFAULT '',
                external_context_json TEXT DEFAULT '{}',
                spoiler_scope TEXT DEFAULT '',
                input_mode TEXT DEFAULT 'text',
                appreciation_stage TEXT DEFAULT '',
                initial_note TEXT NOT NULL,
                free_text TEXT DEFAULT '',
                audio_transcript TEXT DEFAULT '',
                audio_segments_json TEXT DEFAULT '[]',
                questions_json TEXT NOT NULL,
                answers_json TEXT NOT NULL,
                memo_json TEXT NOT NULL,
                edited_memo_json TEXT,
                provider TEXT NOT NULL,
                FOREIGN KEY(work_id) REFERENCES scrapbook_works(id) ON DELETE SET NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS record_embeddings (
                record_id INTEGER PRIMARY KEY,
                model TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(record_id) REFERENCES scrapbook_records(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS record_assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER NOT NULL,
                asset_type TEXT NOT NULL,
                path TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(record_id) REFERENCES scrapbook_records(id) ON DELETE CASCADE
            )
            """
        )
        _ensure_similarity_reasons_table(connection)
        _ensure_record_versions_table(connection)
        _migrate_schema(connection)


def _ensure_record_versions_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS record_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            version_label TEXT NOT NULL,
            record_created_at TEXT NOT NULL,
            record_updated_at TEXT NOT NULL,
            content_type TEXT NOT NULL,
            title TEXT NOT NULL,
            creator TEXT,
            appreciation_date TEXT,
            status TEXT NOT NULL,
            work_context TEXT DEFAULT '',
            external_context_json TEXT DEFAULT '{}',
            spoiler_scope TEXT DEFAULT '',
            input_mode TEXT DEFAULT 'text',
            appreciation_stage TEXT DEFAULT '',
            initial_note TEXT NOT NULL,
            free_text TEXT DEFAULT '',
            audio_transcript TEXT DEFAULT '',
            audio_segments_json TEXT DEFAULT '[]',
            questions_json TEXT NOT NULL,
            answers_json TEXT NOT NULL,
            memo_json TEXT NOT NULL,
            edited_memo_json TEXT,
            provider TEXT NOT NULL,
            FOREIGN KEY(record_id) REFERENCES scrapbook_records(id) ON DELETE CASCADE
        )
        """
    )


def _ensure_similarity_reasons_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS record_similarity_reasons (
            source_record_id INTEGER NOT NULL,
            target_record_id INTEGER NOT NULL,
            source_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            provider TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (
                source_record_id,
                target_record_id,
                source_hash,
                target_hash,
                provider
            ),
            FOREIGN KEY(source_record_id) REFERENCES scrapbook_records(id) ON DELETE CASCADE,
            FOREIGN KEY(target_record_id) REFERENCES scrapbook_records(id) ON DELETE CASCADE
        )
        """
    )


def _migrate_schema(connection: sqlite3.Connection) -> None:
    _ensure_record_versions_table(connection)
    _ensure_similarity_reasons_table(connection)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS scrapbook_works (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            title TEXT NOT NULL,
            content_type TEXT NOT NULL,
            creator TEXT,
            status TEXT NOT NULL
        )
        """
    )
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(scrapbook_records)").fetchall()
    }
    migrations = {
        "work_id": "ALTER TABLE scrapbook_records ADD COLUMN work_id INTEGER",
        "updated_at": "ALTER TABLE scrapbook_records ADD COLUMN updated_at TEXT",
        "appreciation_date": "ALTER TABLE scrapbook_records ADD COLUMN appreciation_date TEXT",
        "input_mode": "ALTER TABLE scrapbook_records ADD COLUMN input_mode TEXT DEFAULT 'text'",
        "appreciation_stage": "ALTER TABLE scrapbook_records ADD COLUMN appreciation_stage TEXT DEFAULT ''",
        "work_context": "ALTER TABLE scrapbook_records ADD COLUMN work_context TEXT DEFAULT ''",
        "external_context_json": "ALTER TABLE scrapbook_records ADD COLUMN external_context_json TEXT DEFAULT '{}'",
        "spoiler_scope": "ALTER TABLE scrapbook_records ADD COLUMN spoiler_scope TEXT DEFAULT ''",
        "free_text": "ALTER TABLE scrapbook_records ADD COLUMN free_text TEXT DEFAULT ''",
        "audio_transcript": "ALTER TABLE scrapbook_records ADD COLUMN audio_transcript TEXT DEFAULT ''",
        "audio_segments_json": "ALTER TABLE scrapbook_records ADD COLUMN audio_segments_json TEXT DEFAULT '[]'",
        "edited_memo_json": "ALTER TABLE scrapbook_records ADD COLUMN edited_memo_json TEXT",
    }
    for column, sql in migrations.items():
        if column not in columns:
            connection.execute(sql)

    connection.execute(
        """
        UPDATE scrapbook_records
        SET updated_at = created_at
        WHERE updated_at IS NULL OR updated_at = ''
        """
    )
    _remove_feedback_columns(connection)
    _backfill_works(connection)


def _remove_feedback_columns(connection: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(scrapbook_records)").fetchall()
    }
    feedback_columns = {
        "feedback_fit",
        "feedback_save",
        "feedback_questions",
        "feedback_ready",
        "feedback_note",
    }
    if not columns.intersection(feedback_columns):
        return

    connection.execute("ALTER TABLE scrapbook_records RENAME TO scrapbook_records_old")
    connection.execute(
        """
        CREATE TABLE scrapbook_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_id INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            content_type TEXT NOT NULL,
            title TEXT NOT NULL,
            creator TEXT,
            appreciation_date TEXT,
            status TEXT NOT NULL,
            work_context TEXT DEFAULT '',
            external_context_json TEXT DEFAULT '{}',
            spoiler_scope TEXT DEFAULT '',
            input_mode TEXT DEFAULT 'text',
            appreciation_stage TEXT DEFAULT '',
            initial_note TEXT NOT NULL,
            free_text TEXT DEFAULT '',
            audio_transcript TEXT DEFAULT '',
            audio_segments_json TEXT DEFAULT '[]',
            questions_json TEXT NOT NULL,
            answers_json TEXT NOT NULL,
            memo_json TEXT NOT NULL,
            edited_memo_json TEXT,
            provider TEXT NOT NULL,
            FOREIGN KEY(work_id) REFERENCES scrapbook_works(id) ON DELETE SET NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO scrapbook_records (
            id,
            work_id,
            created_at,
            updated_at,
            content_type,
            title,
            creator,
            appreciation_date,
            status,
            work_context,
            external_context_json,
            spoiler_scope,
            input_mode,
            appreciation_stage,
            initial_note,
            free_text,
            audio_transcript,
            audio_segments_json,
            questions_json,
            answers_json,
            memo_json,
            edited_memo_json,
            provider
        )
        SELECT
            id,
            work_id,
            created_at,
            COALESCE(updated_at, created_at),
            content_type,
            title,
            creator,
            appreciation_date,
            status,
            COALESCE(work_context, ''),
            COALESCE(external_context_json, '{}'),
            COALESCE(spoiler_scope, ''),
            COALESCE(input_mode, 'text'),
            COALESCE(appreciation_stage, ''),
            initial_note,
            COALESCE(free_text, ''),
            COALESCE(audio_transcript, ''),
            COALESCE(audio_segments_json, '[]'),
            questions_json,
            answers_json,
            memo_json,
            edited_memo_json,
            provider
        FROM scrapbook_records_old
        """
    )
    connection.execute("DROP TABLE scrapbook_records_old")


def _backfill_works(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT id, created_at, updated_at, title, content_type, creator, status
        FROM scrapbook_records
        WHERE work_id IS NULL
        ORDER BY created_at ASC, id ASC
        """
    ).fetchall()
    for row in rows:
        work_id = _find_or_create_work_for_values(
            connection,
            title=row["title"],
            content_type=row["content_type"],
            creator=row["creator"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        connection.execute(
            "UPDATE scrapbook_records SET work_id = ? WHERE id = ?",
            (work_id, row["id"]),
        )


def _find_or_create_work_for_values(
    connection: sqlite3.Connection,
    *,
    title: str,
    content_type: str,
    creator: str | None,
    status: str,
    created_at: str,
    updated_at: str,
) -> int:
    row = connection.execute(
        """
        SELECT id
        FROM scrapbook_works
        WHERE lower(trim(title)) = lower(trim(?))
          AND content_type = ?
          AND COALESCE(creator, '') = COALESCE(?, '')
        ORDER BY id ASC
        LIMIT 1
        """,
        (title, content_type, creator),
    ).fetchone()
    if row:
        connection.execute(
            """
            UPDATE scrapbook_works
            SET updated_at = CASE WHEN updated_at > ? THEN updated_at ELSE ? END,
                status = ?
            WHERE id = ?
            """,
            (updated_at, updated_at, status, row["id"]),
        )
        return int(row["id"])

    cursor = connection.execute(
        """
        INSERT INTO scrapbook_works (
            created_at, updated_at, title, content_type, creator, status
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (created_at, updated_at, title, content_type, creator, status),
    )
    return int(cursor.lastrowid)


def _create_work_for_values(
    connection: sqlite3.Connection,
    *,
    title: str,
    content_type: str,
    creator: str | None,
    status: str,
    created_at: str,
    updated_at: str,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO scrapbook_works (
            created_at, updated_at, title, content_type, creator, status
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (created_at, updated_at, title, content_type, creator, status),
    )
    return int(cursor.lastrowid)


def _latest_record_row_for_work(
    connection: sqlite3.Connection,
    work_id: int,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT *
        FROM scrapbook_records
        WHERE work_id = ?
        ORDER BY updated_at DESC, id DESC
        LIMIT 1
        """,
        (work_id,),
    ).fetchone()


def _load_json_value(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _merge_questions(existing_raw: str | None, incoming: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for question in [*_load_json_value(existing_raw, []), *incoming]:
        text = str(question).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def _merge_answers(existing_raw: str | None, incoming: dict[str, str]) -> dict[str, str]:
    existing = _load_json_value(existing_raw, {})
    if not isinstance(existing, dict):
        existing = {}
    merged = {str(key): str(value) for key, value in existing.items()}
    for key, value in incoming.items():
        text = str(value).strip()
        if text:
            merged[str(key)] = text
    return merged


def _append_text_section(existing: str | None, incoming: str | None, *, label: str) -> str:
    existing_text = str(existing or "").strip()
    incoming_text = str(incoming or "").strip()
    if not incoming_text:
        return existing_text
    if not existing_text:
        return incoming_text
    if incoming_text in existing_text:
        return existing_text
    return f"{existing_text}\n\n[{label}]\n{incoming_text}"


def _has_external_context(payload: dict[str, Any] | None) -> bool:
    return isinstance(payload, dict) and bool(str(payload.get("extract") or "").strip())


def _external_context_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    return payload if _has_external_context(payload) else {}


def _merge_external_context(
    existing_raw: str | None,
    incoming: dict[str, Any] | None,
) -> dict[str, Any]:
    incoming_payload = _external_context_payload(incoming)
    if incoming_payload:
        return incoming_payload
    existing = _load_json_value(existing_raw, {})
    return existing if isinstance(existing, dict) else {}


def _merge_audio_segments(
    existing_raw: str | None,
    incoming: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    existing = _load_json_value(existing_raw, [])
    if not isinstance(existing, list):
        existing = []
    return [*existing, *(incoming or [])]


def _snapshot_record_version(
    connection: sqlite3.Connection,
    *,
    record: sqlite3.Row,
    created_at: str,
    version_label: str,
) -> None:
    connection.execute(
        """
        INSERT INTO record_versions (
            record_id,
            created_at,
            version_label,
            record_created_at,
            record_updated_at,
            content_type,
            title,
            creator,
            appreciation_date,
            status,
            work_context,
            external_context_json,
            spoiler_scope,
            input_mode,
            appreciation_stage,
            initial_note,
            free_text,
            audio_transcript,
            audio_segments_json,
            questions_json,
            answers_json,
            memo_json,
            edited_memo_json,
            provider
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(record["id"]),
            created_at,
            version_label,
            record["created_at"],
            record["updated_at"],
            record["content_type"],
            record["title"],
            record["creator"],
            record["appreciation_date"],
            record["status"],
            record["work_context"] or "",
            record["external_context_json"] or "{}",
            record["spoiler_scope"] or "",
            record["input_mode"] or "text",
            record["appreciation_stage"] or "",
            record["initial_note"],
            record["free_text"] or "",
            record["audio_transcript"] or "",
            record["audio_segments_json"] or "[]",
            record["questions_json"],
            record["answers_json"],
            record["memo_json"],
            record["edited_memo_json"],
            record["provider"],
        ),
    )


def save_record(
    *,
    reflection_input: ReflectionInput,
    questions: list[str],
    answers: dict[str, str],
    memo: dict[str, Any],
    edited_memo: dict[str, Any] | None,
    provider: str,
    work_id: int | None = None,
    force_new_work: bool = False,
    input_mode: str = "text",
    appreciation_stage: str = "",
    audio_segments: list[dict[str, Any]] | None = None,
    external_context: dict[str, Any] | None = None,
) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with open_connection() as connection:
        if work_id:
            resolved_work_id = work_id
        elif force_new_work:
            resolved_work_id = _create_work_for_values(
                connection,
                title=reflection_input.title,
                content_type=reflection_input.content_type,
                creator=reflection_input.creator,
                status=reflection_input.status,
                created_at=now,
                updated_at=now,
            )
        else:
            resolved_work_id = _find_or_create_work_for_values(
                connection,
                title=reflection_input.title,
                content_type=reflection_input.content_type,
                creator=reflection_input.creator,
                status=reflection_input.status,
                created_at=now,
                updated_at=now,
            )
        existing_record = _latest_record_row_for_work(connection, resolved_work_id)
        if existing_record is not None:
            record_id = int(existing_record["id"])
            continuation_label = f"{reflection_input.appreciation_date or now[:10]} 이어 기록"
            _snapshot_record_version(
                connection,
                record=existing_record,
                created_at=now,
                version_label=f"{continuation_label} 전",
            )
            merged_questions = _merge_questions(existing_record["questions_json"], questions)
            merged_answers = _merge_answers(existing_record["answers_json"], answers)
            merged_audio_segments = _merge_audio_segments(
                existing_record["audio_segments_json"],
                audio_segments,
            )
            merged_external_context = _merge_external_context(
                existing_record["external_context_json"],
                external_context,
            )
            connection.execute(
                """
                UPDATE scrapbook_records
                SET updated_at = ?,
                    content_type = ?,
                    title = ?,
                    creator = ?,
                    appreciation_date = ?,
                    status = ?,
                    work_context = ?,
                    external_context_json = ?,
                    spoiler_scope = ?,
                    input_mode = ?,
                    appreciation_stage = ?,
                    initial_note = ?,
                    free_text = ?,
                    audio_transcript = ?,
                    audio_segments_json = ?,
                    questions_json = ?,
                    answers_json = ?,
                    memo_json = ?,
                    edited_memo_json = ?,
                    provider = ?
                WHERE id = ?
                """,
                (
                    now,
                    reflection_input.content_type,
                    reflection_input.title,
                    reflection_input.creator,
                    reflection_input.appreciation_date,
                    reflection_input.status,
                    _append_text_section(
                        existing_record["work_context"],
                        reflection_input.work_context,
                        label=continuation_label,
                    ),
                    json.dumps(merged_external_context, ensure_ascii=False),
                    reflection_input.spoiler_scope,
                    input_mode,
                    appreciation_stage,
                    _append_text_section(
                        existing_record["initial_note"],
                        reflection_input.initial_note,
                        label=continuation_label,
                    ),
                    _append_text_section(
                        existing_record["free_text"],
                        reflection_input.free_text,
                        label=continuation_label,
                    ),
                    _append_text_section(
                        existing_record["audio_transcript"],
                        reflection_input.audio_transcript,
                        label=continuation_label,
                    ),
                    json.dumps(merged_audio_segments, ensure_ascii=False),
                    json.dumps(merged_questions, ensure_ascii=False),
                    json.dumps(merged_answers, ensure_ascii=False),
                    json.dumps(memo, ensure_ascii=False),
                    json.dumps(edited_memo, ensure_ascii=False) if edited_memo else None,
                    provider,
                    record_id,
                ),
            )
            connection.execute(
                """
                UPDATE scrapbook_works
                SET updated_at = ?,
                    content_type = ?,
                    creator = ?,
                    status = ?
                WHERE id = ?
                """,
                (
                    now,
                    reflection_input.content_type,
                    reflection_input.creator,
                    reflection_input.status,
                    resolved_work_id,
                ),
            )
            return record_id
        cursor = connection.execute(
            """
            INSERT INTO scrapbook_records (
                work_id,
                created_at,
                updated_at,
                content_type,
                title,
                creator,
                appreciation_date,
                status,
                work_context,
                external_context_json,
                spoiler_scope,
                input_mode,
                appreciation_stage,
                initial_note,
                free_text,
                audio_transcript,
                audio_segments_json,
                questions_json,
                answers_json,
                memo_json,
                edited_memo_json,
                provider
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_work_id,
                now,
                now,
                reflection_input.content_type,
                reflection_input.title,
                reflection_input.creator,
                reflection_input.appreciation_date,
                reflection_input.status,
                reflection_input.work_context,
                json.dumps(_external_context_payload(external_context), ensure_ascii=False),
                reflection_input.spoiler_scope,
                input_mode,
                appreciation_stage,
                reflection_input.initial_note,
                reflection_input.free_text,
                reflection_input.audio_transcript,
                json.dumps(audio_segments or [], ensure_ascii=False),
                json.dumps(questions, ensure_ascii=False),
                json.dumps(answers, ensure_ascii=False),
                json.dumps(memo, ensure_ascii=False),
                json.dumps(edited_memo, ensure_ascii=False) if edited_memo else None,
                provider,
            ),
        )
        connection.execute(
            """
            UPDATE scrapbook_works
            SET updated_at = ?,
                content_type = ?,
                creator = ?,
                status = ?
            WHERE id = ?
            """,
            (
                now,
                reflection_input.content_type,
                reflection_input.creator,
                reflection_input.status,
                resolved_work_id,
            ),
        )
        return int(cursor.lastrowid)


def update_record(
    *,
    record_id: int,
    edited_memo: dict[str, Any],
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with open_connection() as connection:
        connection.execute(
            """
            UPDATE scrapbook_records
            SET updated_at = ?,
                edited_memo_json = ?
            WHERE id = ?
            """,
            (
                now,
                json.dumps(edited_memo, ensure_ascii=False),
                record_id,
            ),
        )
        connection.execute(
            """
            UPDATE scrapbook_works
            SET updated_at = ?
            WHERE id = (
                SELECT work_id FROM scrapbook_records WHERE id = ?
            )
            """,
            (now, record_id),
        )


def delete_record(record_id: int) -> dict[str, Any] | None:
    with open_connection() as connection:
        record = connection.execute(
            """
            SELECT id, work_id, title
            FROM scrapbook_records
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()
        if record is None:
            return None

        work_id = int(record["work_id"]) if record["work_id"] is not None else None
        connection.execute("DELETE FROM record_assets WHERE record_id = ?", (record_id,))
        connection.execute("DELETE FROM record_embeddings WHERE record_id = ?", (record_id,))
        connection.execute(
            """
            DELETE FROM record_similarity_reasons
            WHERE source_record_id = ? OR target_record_id = ?
            """,
            (record_id, record_id),
        )
        connection.execute("DELETE FROM scrapbook_records WHERE id = ?", (record_id,))

        deleted_work = False
        if work_id is not None:
            remaining = connection.execute(
                "SELECT COUNT(*) AS count FROM scrapbook_records WHERE work_id = ?",
                (work_id,),
            ).fetchone()
            if int(remaining["count"]) == 0:
                connection.execute("DELETE FROM scrapbook_works WHERE id = ?", (work_id,))
                deleted_work = True
            else:
                latest = connection.execute(
                    """
                    SELECT updated_at, content_type, creator, status
                    FROM scrapbook_records
                    WHERE work_id = ?
                    ORDER BY updated_at DESC, id DESC
                    LIMIT 1
                    """,
                    (work_id,),
                ).fetchone()
                if latest is not None:
                    connection.execute(
                        """
                        UPDATE scrapbook_works
                        SET updated_at = ?,
                            content_type = ?,
                            creator = ?,
                            status = ?
                        WHERE id = ?
                        """,
                        (
                            latest["updated_at"],
                            latest["content_type"],
                            latest["creator"],
                            latest["status"],
                            work_id,
                        ),
                    )

        return {
            "record_id": int(record["id"]),
            "work_id": work_id,
            "title": record["title"],
            "deleted_work": deleted_work,
        }


def get_work(work_id: int) -> dict[str, Any] | None:
    with open_connection() as connection:
        row = connection.execute(
            """
            SELECT
                w.*,
                COUNT(r.id) AS record_count,
                MAX(r.updated_at) AS latest_record_updated_at
            FROM scrapbook_works w
            LEFT JOIN scrapbook_records r ON r.work_id = w.id
            WHERE w.id = ?
            GROUP BY w.id
            """,
            (work_id,),
        ).fetchone()
        if row is None:
            return None
        work = dict(row)
        latest = connection.execute(
            """
            SELECT id
            FROM scrapbook_records
            WHERE work_id = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (work_id,),
        ).fetchone()
    work["latest_record_id"] = int(latest["id"]) if latest else None
    return work


def list_works(
    search_query: str = "",
    *,
    content_type: str | list[str] = "전체",
    status: str | list[str] = "전체",
    sort_by: str = "recent_updated",
) -> list[dict[str, Any]]:
    search_query = search_query.strip()
    where_clauses: list[str] = []
    params: list[Any] = []

    if search_query:
        like = f"%{search_query}%"
        where_clauses.append(
            """
            (
                w.title LIKE ?
                OR w.content_type LIKE ?
                OR EXISTS (
                    SELECT 1
                    FROM scrapbook_records sr
                    WHERE sr.work_id = w.id
                      AND (
                        sr.initial_note LIKE ?
                        OR sr.work_context LIKE ?
                        OR sr.spoiler_scope LIKE ?
                        OR sr.free_text LIKE ?
                        OR sr.audio_transcript LIKE ?
                        OR sr.questions_json LIKE ?
                        OR sr.answers_json LIKE ?
                        OR sr.memo_json LIKE ?
                        OR COALESCE(sr.edited_memo_json, '') LIKE ?
                      )
                )
            )
            """
        )
        params.extend([like, like, like, like, like, like, like, like, like, like, like])

    content_type_values = [content_type] if isinstance(content_type, str) else content_type
    content_type_values = [
        value for value in content_type_values if value and value != "전체"
    ]
    if content_type_values:
        placeholders = ", ".join("?" for _ in content_type_values)
        where_clauses.append(f"w.content_type IN ({placeholders})")
        params.extend(content_type_values)

    status_values = [status] if isinstance(status, str) else status
    status_values = [value for value in status_values if value and value != "전체"]
    if status_values:
        placeholders = ", ".join("?" for _ in status_values)
        where_clauses.append(f"w.status IN ({placeholders})")
        params.extend(status_values)

    order_by_options = {
        "recent_updated": "w.updated_at DESC, w.id DESC",
        "recent_created": "w.created_at DESC, w.id DESC",
        "appreciation_date": "latest_appreciation_date DESC, w.updated_at DESC, w.id DESC",
    }
    order_by = order_by_options.get(sort_by, order_by_options["recent_updated"])
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with open_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT
                w.id,
                w.created_at,
                w.updated_at,
                w.title,
                w.content_type,
                w.creator,
                w.status,
                COUNT(r.id) AS record_count,
                MAX(COALESCE(r.appreciation_date, r.created_at)) AS latest_appreciation_date
            FROM scrapbook_works w
            LEFT JOIN scrapbook_records r ON r.work_id = w.id
            {where_sql}
            GROUP BY w.id
            ORDER BY {order_by}
            """,
            params,
        ).fetchall()

        works = []
        for row in rows:
            work = dict(row)
            latest = connection.execute(
                """
                SELECT id
                FROM scrapbook_records
                WHERE work_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (work["id"],),
            ).fetchone()
            work["latest_record_id"] = int(latest["id"]) if latest else None
            works.append(work)
    return works


def get_work_records(work_id: int, *, ascending: bool = False) -> list[dict[str, Any]]:
    direction = "ASC" if ascending else "DESC"
    with open_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT id
            FROM scrapbook_records
            WHERE work_id = ?
            ORDER BY COALESCE(appreciation_date, created_at) {direction}, id {direction}
            """,
            (work_id,),
        ).fetchall()
    return [record for row in rows if (record := get_record(int(row["id"])))]


def find_matching_works(title: str, content_type: str | None = None) -> list[dict[str, Any]]:
    title = title.strip()
    if not title:
        return []
    where = "WHERE lower(trim(title)) = lower(trim(?))"
    params: list[Any] = [title]
    if content_type:
        where += " AND content_type = ?"
        params.append(content_type)
    with open_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT id
            FROM scrapbook_works
            {where}
            ORDER BY updated_at DESC, id DESC
            """,
            params,
        ).fetchall()
    return [work for row in rows if (work := get_work(int(row["id"])))]


def list_records(
    search_query: str = "",
    *,
    content_type: str | list[str] = "전체",
    status: str | list[str] = "전체",
    sort_by: str = "recent_updated",
) -> list[dict[str, Any]]:
    search_query = search_query.strip()
    where_clauses: list[str] = []
    params: list[Any] = []

    if search_query:
        like = f"%{search_query}%"
        where_clauses.append(
            """
            (
                title LIKE ?
                OR content_type LIKE ?
                OR initial_note LIKE ?
                OR work_context LIKE ?
                OR spoiler_scope LIKE ?
                OR free_text LIKE ?
                OR audio_transcript LIKE ?
                OR questions_json LIKE ?
                OR answers_json LIKE ?
                OR memo_json LIKE ?
                OR COALESCE(edited_memo_json, '') LIKE ?
            )
            """
        )
        params.extend([like, like, like, like, like, like, like, like, like, like, like])

    content_type_values = [content_type] if isinstance(content_type, str) else content_type
    content_type_values = [
        value for value in content_type_values if value and value != "전체"
    ]
    if content_type_values:
        placeholders = ", ".join("?" for _ in content_type_values)
        where_clauses.append(f"content_type IN ({placeholders})")
        params.extend(content_type_values)

    status_values = [status] if isinstance(status, str) else status
    status_values = [value for value in status_values if value and value != "전체"]
    if status_values:
        placeholders = ", ".join("?" for _ in status_values)
        where_clauses.append(f"status IN ({placeholders})")
        params.extend(status_values)

    order_by_options = {
        "recent_updated": "updated_at DESC, id DESC",
        "recent_created": "created_at DESC, id DESC",
        "appreciation_date": "COALESCE(appreciation_date, created_at) DESC, id DESC",
    }
    order_by = order_by_options.get(sort_by, order_by_options["recent_updated"])
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with open_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT id, work_id, created_at, updated_at, content_type, title, creator,
                   appreciation_date, status, work_context, spoiler_scope, input_mode, appreciation_stage, provider
            FROM scrapbook_records
            {where_sql}
            ORDER BY {order_by}
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_record(record_id: int) -> dict[str, Any] | None:
    with open_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM scrapbook_records
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()

    if row is None:
        return None

    record = dict(row)
    record["questions"] = json.loads(record.pop("questions_json"))
    record["answers"] = json.loads(record.pop("answers_json"))
    record["memo"] = json.loads(record.pop("memo_json"))
    audio_segments_json = record.pop("audio_segments_json", "[]")
    record["audio_segments"] = json.loads(audio_segments_json or "[]")
    external_context_json = record.pop("external_context_json", "{}")
    try:
        record["external_context"] = json.loads(external_context_json or "{}")
    except json.JSONDecodeError:
        record["external_context"] = {}
    edited_memo_json = record.pop("edited_memo_json", None)
    record["edited_memo"] = json.loads(edited_memo_json) if edited_memo_json else None
    return record


def _decode_record_version(row: sqlite3.Row) -> dict[str, Any]:
    version = dict(row)
    version["questions"] = json.loads(version.pop("questions_json"))
    version["answers"] = json.loads(version.pop("answers_json"))
    version["memo"] = json.loads(version.pop("memo_json"))
    audio_segments_json = version.pop("audio_segments_json", "[]")
    version["audio_segments"] = json.loads(audio_segments_json or "[]")
    external_context_json = version.pop("external_context_json", "{}")
    try:
        version["external_context"] = json.loads(external_context_json or "{}")
    except json.JSONDecodeError:
        version["external_context"] = {}
    edited_memo_json = version.pop("edited_memo_json", None)
    version["edited_memo"] = json.loads(edited_memo_json) if edited_memo_json else None
    return version


def list_record_versions(record_id: int) -> list[dict[str, Any]]:
    with open_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM record_versions
            WHERE record_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (record_id,),
        ).fetchall()
    return [_decode_record_version(row) for row in rows]


def upsert_record_embedding(
    *,
    record_id: int,
    model: str,
    embedding: list[float],
    source_hash: str,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with open_connection() as connection:
        connection.execute(
            """
            INSERT INTO record_embeddings (
                record_id,
                model,
                embedding_json,
                source_hash,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                model = excluded.model,
                embedding_json = excluded.embedding_json,
                source_hash = excluded.source_hash,
                updated_at = excluded.updated_at
            """,
            (
                record_id,
                model,
                json.dumps(embedding),
                source_hash,
                now,
                now,
            ),
        )
        connection.execute(
            """
            DELETE FROM record_similarity_reasons
            WHERE (source_record_id = ? AND source_hash <> ?)
               OR (target_record_id = ? AND target_hash <> ?)
            """,
            (record_id, source_hash, record_id, source_hash),
        )


def get_record_embedding(record_id: int) -> dict[str, Any] | None:
    with open_connection() as connection:
        row = connection.execute(
            """
            SELECT record_id, model, embedding_json, source_hash, created_at, updated_at
            FROM record_embeddings
            WHERE record_id = ?
            """,
            (record_id,),
        ).fetchone()
    if row is None:
        return None
    embedding = dict(row)
    embedding["embedding"] = json.loads(embedding.pop("embedding_json"))
    return embedding


def list_record_embeddings() -> list[dict[str, Any]]:
    with open_connection() as connection:
        rows = connection.execute(
            """
            SELECT record_id, model, embedding_json, source_hash, created_at, updated_at
            FROM record_embeddings
            """
        ).fetchall()
    embeddings = []
    for row in rows:
        item = dict(row)
        item["embedding"] = json.loads(item.pop("embedding_json"))
        embeddings.append(item)
    return embeddings


def get_similarity_reason(
    *,
    source_record_id: int,
    target_record_id: int,
    source_hash: str,
    target_hash: str,
    provider: str,
) -> str | None:
    with open_connection() as connection:
        row = connection.execute(
            """
            SELECT reason
            FROM record_similarity_reasons
            WHERE source_record_id = ?
              AND target_record_id = ?
              AND source_hash = ?
              AND target_hash = ?
              AND provider = ?
            """,
            (
                source_record_id,
                target_record_id,
                source_hash,
                target_hash,
                provider,
            ),
        ).fetchone()
    if row is None:
        return None
    return str(row["reason"])


def upsert_similarity_reason(
    *,
    source_record_id: int,
    target_record_id: int,
    source_hash: str,
    target_hash: str,
    provider: str,
    reason: str,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with open_connection() as connection:
        connection.execute(
            """
            INSERT INTO record_similarity_reasons (
                source_record_id,
                target_record_id,
                source_hash,
                target_hash,
                provider,
                reason,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                source_record_id,
                target_record_id,
                source_hash,
                target_hash,
                provider
            ) DO UPDATE SET
                reason = excluded.reason,
                updated_at = excluded.updated_at
            """,
            (
                source_record_id,
                target_record_id,
                source_hash,
                target_hash,
                provider,
                reason,
                now,
                now,
            ),
        )


def add_record_asset(
    *,
    record_id: int,
    asset_type: str,
    path: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with open_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO record_assets (
                record_id,
                asset_type,
                path,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record_id,
                asset_type,
                path,
                json.dumps(metadata or {}, ensure_ascii=False),
                now,
            ),
        )
        return int(cursor.lastrowid)


def list_record_assets(
    record_id: int,
    *,
    asset_type: str | None = None,
) -> list[dict[str, Any]]:
    where = "WHERE record_id = ?"
    params: list[Any] = [record_id]
    if asset_type:
        where += " AND asset_type = ?"
        params.append(asset_type)

    with open_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT id, record_id, asset_type, path, metadata_json, created_at
            FROM record_assets
            {where}
            ORDER BY created_at DESC, id DESC
            """,
            params,
        ).fetchall()

    assets = []
    for row in rows:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        assets.append(item)
    return assets
