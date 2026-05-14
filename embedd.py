from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path
from typing import Any

from utils.database import close_db_pools, exec_query, start_pooling
from utils.embeddings import embed_text, vector_literal

# Keep dimension aligned with existing schema in scripts/bootstrap_supabase_schema.py
EMBEDDING_DIM = 1536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create student_knowledge table, embed user_intent, and push CSV data to database."
    )
    parser.add_argument(
        "--csv",
        default="design/student_case_dataset.csv",
        help="Path to source CSV file.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate table before inserting data.",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        non_empty_lines = (line for line in f if line.strip())
        reader = csv.DictReader(non_empty_lines)
        expected_cols = {
            "user_intent",
            "response_mode",
            "should_do",
            "should_not_do",
            "sample_responses",
        }
        missing = expected_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        rows = []
        for row in reader:
            clean_row = {k: (v or "").strip() for k, v in row.items()}
            if clean_row["user_intent"]:
                rows.append(clean_row)
        return rows


async def ensure_schema() -> None:
    await exec_query("CREATE EXTENSION IF NOT EXISTS vector")

    await exec_query(
        f"""
        CREATE TABLE IF NOT EXISTS student_knowledge (
            id BIGSERIAL PRIMARY KEY,
            user_intent TEXT NOT NULL,
            response_mode TEXT,
            should_do TEXT,
            should_not_do TEXT,
            sample_responses TEXT,
            embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )

    await exec_query(
        """
        CREATE INDEX IF NOT EXISTS idx_student_knowledge_embedding
        ON student_knowledge USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
        """
    )


async def insert_rows(rows: list[dict[str, str]], truncate: bool) -> None:
    if truncate:
        await exec_query("TRUNCATE TABLE student_knowledge RESTART IDENTITY")

    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        embedding = await embed_text(row["user_intent"])
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}"
            )

        await exec_query(
            """
            INSERT INTO student_knowledge (
                user_intent,
                response_mode,
                should_do,
                should_not_do,
                sample_responses,
                embedding
            )
            VALUES ($1, $2, $3, $4, $5, $6::vector)
            """,
            row["user_intent"],
            row["response_mode"],
            row["should_do"],
            row["should_not_do"],
            row["sample_responses"],
            vector_literal(embedding),
        )

        if idx % 25 == 0 or idx == total:
            print(f"Inserted {idx}/{total} rows")


async def run(csv_path: Path, truncate: bool) -> dict[str, Any]:
    rows = read_rows(csv_path)
    if not rows:
        return {"status": "no_data", "inserted": 0}

    await start_pooling()
    try:
        await ensure_schema()
        await insert_rows(rows, truncate=truncate)
        return {"status": "ok", "inserted": len(rows)}
    finally:
        await close_db_pools()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    result = asyncio.run(run(csv_path=csv_path, truncate=args.truncate))
    print(result)


if __name__ == "__main__":
    main()
