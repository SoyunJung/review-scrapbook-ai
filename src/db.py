from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from src.config import DATA_DIR, DB_PATH
from src.models import ReflectionInput


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scrapbook_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                content_type TEXT NOT NULL,
                title TEXT NOT NULL,
                creator TEXT,
                status TEXT NOT NULL,
                initial_note TEXT NOT NULL,
                questions_json TEXT NOT NULL,
                answers_json TEXT NOT NULL,
                memo_json TEXT NOT NULL,
                provider TEXT NOT NULL
            )
            """
        )


def save_record(
    *,
    reflection_input: ReflectionInput,
    questions: list[str],
    answers: dict[str, str],
    memo: dict[str, Any],
    provider: str,
) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO scrapbook_records (
                created_at,
                content_type,
                title,
                creator,
                status,
                initial_note,
                questions_json,
                answers_json,
                memo_json,
                provider
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                reflection_input.content_type,
                reflection_input.title,
                reflection_input.creator,
                reflection_input.status,
                reflection_input.initial_note,
                json.dumps(questions, ensure_ascii=False),
                json.dumps(answers, ensure_ascii=False),
                json.dumps(memo, ensure_ascii=False),
                provider,
            ),
        )
        return int(cursor.lastrowid)


def list_records() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, content_type, title, creator, status, provider
            FROM scrapbook_records
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_record(record_id: int) -> dict[str, Any] | None:
    with get_connection() as connection:
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
    return record
