import os, hashlib, tempfile, os as _os, json
from contextlib import asynccontextmanager
import asyncpg
from fastapi import FastAPI, UploadFile, File as F, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.storage import s3, BUCKET, ensure_bucket
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.parser import download_file_to_tmp, parse_pdf_to_rows_combined, parse_pdf_to_rows, md5_row
import re

load_dotenv()

DB_DSN = os.getenv("DATABASE_URL")
assert DB_DSN, "Missing DATABASE_URL"

pg_pool: asyncpg.pool.Pool | None = None
print("LOADED:", __file__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_bucket()
    global pg_pool
    pg_pool = await asyncpg.create_pool(dsn=DB_DSN, min_size=1, max_size=5)
    await ensure_tables()
    try:
        yield
    finally:
        await pg_pool.close()

async def ensure_tables():
    async with pg_pool.acquire() as conn:
        await conn.execute("""
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

        -- One-time cleanup is safe to keep; it’s a no-op when already clean.
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
        """)


async def get_db() -> asyncpg.Connection:
    async with pg_pool.acquire() as conn:
        yield conn

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):3000",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PresignIn(BaseModel):
    sha256: str
    filename: str
    content_type: str = "application/pdf"

@app.post("/upload/presign")
async def presign_put_endpoint(body: PresignIn):
    key = f"{body.sha256}/{body.filename}"
    put_url = s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": BUCKET, "Key": key, "ContentType": body.content_type},
        ExpiresIn=3600,
    )
    return {"url": put_url, "key": key}

class RegisterIn(BaseModel):
    sha256: str
    filename: str
    key: str

@app.post("/files/register")
async def register_endpoint(body: RegisterIn, db: asyncpg.Connection = Depends(get_db)):
    stored_url = f"s3://{BUCKET}/{body.key}"
    row = await db.fetchrow("SELECT id FROM files WHERE sha256=$1", body.sha256)
    if row is None:
        row = await db.fetchrow(
            """INSERT INTO files (name, sha256, pages, stored_url)
               VALUES ($1,$2,$3,$4) RETURNING id""",
            body.filename, body.sha256, None, stored_url
        )
    return {"file_id": str(row["id"]), "sha256": body.sha256, "stored_url": stored_url}

async def audit_table(table_id: str, db: asyncpg.Connection) -> dict:
    stats = await db.fetchrow(
        """
        SELECT
          COUNT(*) AS row_count,
          SUM(amount) AS total_amount,
          SUM(CASE WHEN amount < 0 THEN 1 ELSE 0 END) AS negative_count,
          SUM(CASE WHEN amount IS NULL OR amount = 0 THEN 1 ELSE 0 END) AS zero_or_null_count
        FROM rows
        WHERE table_id = $1::uuid
        """,
        table_id,
    )

    row_count = stats["row_count"] or 0
    total_amount = stats["total_amount"]
    negative_count = stats["negative_count"] or 0
    zero_null_count = stats["zero_or_null_count"] or 0

    messages: list[dict] = []

    if total_amount is None or row_count == 0:
        messages.append(
            {"type": "error", "message": "No usable rows found for this table."}
        )
        passed = False
    else:
        messages.append(
            {
                "type": "info",
                "message": f"Total amount across {row_count} rows: {total_amount}",
            }
        )
        if negative_count:
            messages.append(
                {
                    "type": "warning",
                    "message": f"{negative_count} rows have negative amounts.",
                }
            )
        if zero_null_count:
            messages.append(
                {
                    "type": "warning",
                    "message": f"{zero_null_count} rows have zero or missing amounts.",
                }
            )

        # MVP pass condition: we have some rows and a total
        passed = True

    await db.execute(
        "INSERT INTO audits(table_id, passed, messages) VALUES ($1::uuid, $2, $3)",
        table_id,
        passed,
        json.dumps(messages),
    )

    return {"passed": passed, "messages": messages}


class ParseIn(BaseModel):
    file_id: str
    table_label: str

@app.post("/parse")
async def parse_pdf_endpoint(body: ParseIn, db=Depends(get_db)):
    try:
        frow = await db.fetchrow("SELECT id, stored_url FROM files WHERE id=$1::uuid", body.file_id)
        if not frow:
            raise HTTPException(404, "file not found")

        trow = await db.fetchrow(
            """INSERT INTO tables (file_id, label, parser_version)
               VALUES ($1,$2,$3) RETURNING id""",
            frow["id"], body.table_label, "v0.1"
        )
        table_id = trow["id"]

        tmp = download_file_to_tmp(frow["stored_url"])
        try:
            rows = parse_pdf_to_rows_combined(tmp)
        finally:
            try: os.remove(tmp)
            except: pass

        clean_rows = []
        for r in rows:
            name = r["program_name"]
            if len(name) < 4: 
                continue
            # must contain at least 2 alphabetic tokens (avoid “a total d”, “o”, “•”)
            if len([t for t in re.findall(r"[A-Za-z]+", name) if len(t) >= 2]) < 2:
                continue
            if r["amount"] is None:
                continue
            clean_rows.append(r)

        rows = clean_rows

        for r in rows:
            checksum = md5_row(r["program_name"], r["fy"], r["amount"], r["page"])
            bbox_val = r.get("bbox")
            bbox_json = json.dumps(bbox_val) if bbox_val else None
            await db.execute(
                """INSERT INTO rows (table_id, program_id, program_name, fy, amount, bbox, page, checksum)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT (table_id, checksum) DO NOTHING""",
                table_id, None, r["program_name"], r["fy"], r["amount"], bbox_json, r["page"], checksum
            )

        audit = await audit_table(str(table_id), db)
        return {
            "table_id": str(table_id),
            "audit": audit,
            "count": len(rows),
        }

    except HTTPException:
        raise
    except Exception as e:
        # print full error in your uvicorn logs so you can see the stack
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail="parse failed")

@app.get("/tables/{table_id}/audit")
async def get_audit(table_id: str, db=Depends(get_db)):
    row = await db.fetchrow(
        """
        SELECT passed, messages, created_at
        FROM audits
        WHERE table_id = $1::uuid
        ORDER BY created_at DESC
        LIMIT 1
        """,
        table_id,
    )
    if not row:
        raise HTTPException(404, "audit not found")

    return {
        "passed": row["passed"],
        "messages": row["messages"],
        "created_at": row["created_at"].isoformat(),
    }


@app.get("/tables/{table_id}/preview")
async def preview_rows(table_id: str, db=Depends(get_db)):
    rows = await db.fetch(
        """-- replace SELECT in /tables/{table_id}/preview
            SELECT program_name, fy, amount, page
            FROM rows
            WHERE table_id = $1::uuid
            AND length(trim(program_name)) >= 3
            AND program_name !~ '^[\W•.\-–—\s]+$'
            AND amount >= 1000
            ORDER BY page, program_name
            LIMIT 10;""",
        table_id
    )
    return [dict(r) for r in rows]


@app.get("/health")
def health():
    return {"status": "ok"}
