import os
import re
import json
import asyncio
import asyncpg
from decimal import Decimal

from app.storage import s3, BUCKET
from app.parser import (
    download_file_to_tmp,
    parse_pdf_to_rows_combined,
    clean_program_title,
    get_program_id,
    md5_row,
)
from dotenv import load_dotenv

load_dotenv()
DB_DSN = os.getenv("DATABASE_URL")
assert DB_DSN, "Missing DATABASE_URL"


DDL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS files (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  pages INT,
  stored_url TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS files_sha256_idx ON files(sha256);

CREATE TABLE IF NOT EXISTS tables (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  file_id UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  label TEXT NOT NULL,
  page_start INT,
  page_end INT,
  parser_version TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rows (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  table_id UUID NOT NULL REFERENCES tables(id) ON DELETE CASCADE,
  program_id TEXT,
  program_name TEXT,
  fy INT,
  amount BIGINT,
  bbox JSONB,
  page INT,
  checksum TEXT
);

CREATE TABLE IF NOT EXISTS audits (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  table_id UUID NOT NULL REFERENCES tables(id) ON DELETE CASCADE,
  passed BOOLEAN NOT NULL,
  messages JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS diffs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  prev_table_id UUID NOT NULL REFERENCES tables(id) ON DELETE CASCADE,
  curr_table_id UUID NOT NULL REFERENCES tables(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS diff_rows (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  diff_id UUID NOT NULL REFERENCES diffs(id) ON DELETE CASCADE,
  program_name TEXT,
  prev_amount BIGINT,
  curr_amount BIGINT,
  delta_abs BIGINT,
  delta_pct DOUBLE PRECISION
);

WITH d AS (
  SELECT ctid,
         ROW_NUMBER() OVER (PARTITION BY table_id, checksum ORDER BY ctid) AS rn
  FROM rows
  WHERE checksum IS NOT NULL
)
DELETE FROM rows r
USING d
WHERE r.ctid = d.ctid
  AND d.rn > 1;

CREATE UNIQUE INDEX IF NOT EXISTS rows_table_id_checksum_uniq
ON rows(table_id, checksum);
"""

async def ensure_tables_local(pool: asyncpg.pool.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(DDL)


async def backfill_from_minio():
    pool = await asyncpg.create_pool(dsn=DB_DSN, min_size=1, max_size=5)

    await ensure_tables_local(pool)

    async with pool.acquire() as db:
        paginator = s3.get_paginator("list_objects_v2")


        async def ensure_file_table_rows(key: str):
            if not key.lower().endswith(".pdf"):
                return

            parts = key.split("/", 1)
            if len(parts) != 2:
                return

            sha256, filename = parts
            stored_url = f"s3://{BUCKET}/{key}"

            existing = await db.fetchrow(
                "SELECT id FROM files WHERE sha256 = $1",
                sha256,
            )
            if existing:
                print(f"Skipping {key} (already in files)")
                return

            print(f"Ingesting {key}")

            file_row = await db.fetchrow(
                """
                INSERT INTO files (name, sha256, pages, stored_url)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                filename,
                sha256,
                None,
                stored_url,
            )
            file_id = file_row["id"]

            trow = await db.fetchrow(
                """
                INSERT INTO tables (file_id, label, parser_version)
                VALUES ($1, $2, $3)
                RETURNING id
                """,
                file_id,
                filename,
                "v0.1",
            )
            table_id = trow["id"]

            tmp = download_file_to_tmp(stored_url)
            try:
                rows = parse_pdf_to_rows_combined(tmp)
            finally:
                try:
                    os.remove(tmp)
                except FileNotFoundError:
                    pass

            clean_rows = []
            for r in rows:
                raw_name = r.get("program_name") or ""
                name = clean_program_title(raw_name)

                if len(name) < 4:
                    continue

                if len([t for t in re.findall(r"[A-Za-z]+", name) if len(t) >= 2]) < 2:
                    continue

                if r.get("amount") is None:
                    continue

                r["program_name"] = name
                clean_rows.append(r)

            for r in clean_rows:
                name = r["program_name"]
                program_id = get_program_id(name)
                checksum = md5_row(name, r["fy"], r["amount"], r["page"])
                bbox_val = r.get("bbox")
                bbox_json = json.dumps(bbox_val) if bbox_val else None

                await db.execute(
                    """
                    INSERT INTO rows (table_id, program_id, program_name, fy, amount, bbox, page, checksum)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                    ON CONFLICT (table_id, checksum) DO NOTHING
                    """,
                    table_id,
                    program_id,
                    name,
                    r["fy"],
                    r["amount"],
                    bbox_json,
                    r["page"],
                    checksum,
                )

            print(f"Done {key}: {len(clean_rows)} rows")

        # walk the bucket
        for page in paginator.paginate(Bucket=BUCKET):
            for obj in page.get("Contents", []):
                await ensure_file_table_rows(obj["Key"])

    await pool.close()


if __name__ == "__main__":
    asyncio.run(backfill_from_minio())
