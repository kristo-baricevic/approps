import os, hashlib, tempfile, os as _os, json
from contextlib import asynccontextmanager
import asyncpg
from fastapi import FastAPI, UploadFile, File as F, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.storage import s3, BUCKET, ensure_bucket
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.parser import download_file_to_tmp, parse_pdf_to_rows, md5_row

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
            rows = parse_pdf_to_rows(tmp)
        finally:
            try: os.remove(tmp)
            except: pass

        for r in rows:
            checksum = md5_row(r["program_name"], r["fy"], r["amount"], r["page"])
            bbox_val = r.get("bbox")
            bbox_json = json.dumps(bbox_val) if bbox_val else None
            await db.execute(
                """INSERT INTO rows (table_id, program_id, program_name, fy, amount, bbox, page, checksum)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
                table_id, None, r["program_name"], r["fy"], r["amount"], bbox_json, r["page"], checksum
            )

        return {"table_id": str(table_id), "audit": {"passed": True, "messages": []}, "count": len(rows)}

    except HTTPException:
        raise
    except Exception as e:
        # print full error in your uvicorn logs so you can see the stack
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail="parse failed")

@app.get("/tables/{table_id}/preview")
async def preview_rows(table_id: str, db=Depends(get_db)):
    rows = await db.fetch(
        """SELECT program_name, fy, amount, page FROM rows
           WHERE table_id=$1::uuid
           ORDER BY page, program_name
           LIMIT 10""",
        table_id
    )
    return [dict(r) for r in rows]


@app.get("/health")
def health():
    return {"status": "ok"}
