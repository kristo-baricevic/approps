import os, hashlib, tempfile, os as _os, json
from contextlib import asynccontextmanager
import asyncpg
from fastapi import FastAPI, UploadFile, File as F, Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from app.storage import s3, BUCKET, ensure_bucket
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.parser import download_file_to_tmp, get_program_id, ai_label_program, looks_like_program_name, clean_program_title, parse_pdf_to_rows_combined, parse_pdf_to_rows, md5_row, s3_url_to_bucket_key
import re
import pdfplumber
import tempfile
from decimal import Decimal
from typing import Any, List, Optional
import io, csv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from botocore.exceptions import ClientError
from openai import AsyncOpenAI

load_dotenv()

DB_DSN = os.getenv("DATABASE_URL")
assert DB_DSN, "Missing DATABASE_URL"

pg_pool: asyncpg.pool.Pool | None = None
print("LOADED:", __file__)

AI_ENABLED = bool(os.getenv("OPENAI_API_KEY"))
_ai_client = AsyncOpenAI() if AI_ENABLED else None

GENERIC_PROGRAM_RE = re.compile(
    r"""
    ^agreement\b|
    ^includes?\b|
    ^this\b|
    ^within\b|
    ^which\b|
    ^no\s+less\s+than\b|
    ^increase\s+of\b|
    ^ending\b|
    ^support\s+a\b|
    total\ discretionary\ allocation
    """,
    re.I | re.X,
)


def needs_ai_refinement(name: str) -> bool:
    if not name:
        return False
    s = name.strip().lower()
    if len(s.split()) <= 2:
        return True
    if GENERIC_PROGRAM_RE.search(s):
        return True
    return False


async def call_ai_for_program(
    context: str,
    amount: int | None,
    fy: int | None,
    file_label: str,
) -> tuple[str, str]:
    if not AI_ENABLED or not context:
        return "", ""

    system_msg = (
        "You are helping label budget line items from US appropriations explanatory statements. "
        "Given a short excerpt, identify the specific program or activity being funded and write "
        "a short human readable description. Keep the name concise and the description brief."
    )

    user_msg = f"""
Bill label: {file_label}
Amount: {amount if amount is not None else "UNKNOWN"}
Fiscal year: {fy if fy is not None else "UNKNOWN"}

Context:
{context}

Return JSON with keys "name" and "brief". Example:
{{"name": "Behavioral Health Workforce Education and Training", "brief": "Funds training programs that expand the mental and behavioral health workforce."}}
"""

    try:
        resp = await _ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        content = resp.choices[0].message.content or ""
        data = json.loads(content)
        name = (data.get("name") or "").strip()
        brief = (data.get("brief") or "").strip()
        return name, brief
    except Exception:
        return "", ""


def normalize_bbox(raw: Any) -> Optional[List[float]]:
    if raw is None:
        return None

    # If it came back from the DB as a JSON string
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            raw = json.loads(s)
        except json.JSONDecodeError:
            # Try simple "x0,top,x1,bottom" format
            parts = [p.strip() for p in s.split(",")]
            if len(parts) != 4:
                return None
            try:
                return [float(p) for p in parts]
            except ValueError:
                return None

    # If it is a dict like {"x0":..., "top":..., "x1":..., "bottom":...}
    if isinstance(raw, dict):
        keys = ["x0", "top", "x1", "bottom"]
        if not all(k in raw for k in keys):
            return None
        try:
            return [float(raw[k]) for k in keys]
        except (TypeError, ValueError):
            return None

    # If it is already a list/tuple
    if isinstance(raw, (list, tuple)):
        if len(raw) != 4:
            return None
        try:
            return [float(v) if not isinstance(v, Decimal) else float(v) for v in raw]
        except (TypeError, ValueError):
            return None

    return None


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
            program_name TEXT,          -- final label that UI uses
            program_name_raw TEXT,      -- raw heuristic text from parser
            program_ai_name TEXT,       -- AI suggested cleaner name
            program_ai_brief TEXT,      -- AI short description
            program_name_source TEXT,   -- 'heuristic' or 'ai'

            fy INT,
            amount BIGINT,
            bbox JSONB,
            page INT,
            checksum TEXT
        );


        ALTER TABLE rows ADD COLUMN IF NOT EXISTS program_name_raw TEXT;
        ALTER TABLE rows ADD COLUMN IF NOT EXISTS program_ai_name TEXT;
        ALTER TABLE rows ADD COLUMN IF NOT EXISTS program_ai_brief TEXT;
        ALTER TABLE rows ADD COLUMN IF NOT EXISTS program_name_source TEXT;

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
        """)


async def get_db() -> asyncpg.Connection:
    async with pg_pool.acquire() as conn:
        yield conn

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://approps.vercel.app",   # frontend
        "https://api-approps.com",      # api domain (if you ever hit it directly from browser)
    ],
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
    external_url = put_url.replace("http://minio:9000", "https://api-approps.com")
    return {"url": external_url, "key": key}

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


class ParseIn(BaseModel):
    file_id: str
    table_label: str


@app.post("/parse")
async def parse_pdf_endpoint(body: ParseIn, db=Depends(get_db)):
    try:
        # select id, stored_url, name so we can use file_name later
        frow = await db.fetchrow(
            "SELECT id, stored_url, name FROM files WHERE id=$1::uuid",
            body.file_id,
        )
        if not frow:
            raise HTTPException(404, "file not found")

        file_id = frow["id"]
        file_name = frow["name"]
        stored_url = frow["stored_url"]

        trow = await db.fetchrow(
            """INSERT INTO tables (file_id, label, parser_version)
               VALUES ($1,$2,$3) RETURNING id""",
            file_id,
            body.table_label,
            "v0.1",
        )
        table_id = trow["id"]

        tmp = download_file_to_tmp(stored_url)
        try:
            rows = parse_pdf_to_rows_combined(tmp)
        finally:
            try:
                os.remove(tmp)
            except:
                pass

        clean_rows = []

        for r in rows:
            raw_name = r.get("program_name") or ""
            name = clean_program_title(raw_name)

            if len(name) < 4:
                continue

            # must contain at least 2 alphabetic tokens (avoid “a total d”, “o”, “•”, “period”)
            if len([t for t in re.findall(r"[A-Za-z]+", name) if len(t) >= 2]) < 2:
                continue

            if r["amount"] is None:
                continue

            # keep both original and cleaned
            r["program_name_raw"] = raw_name
            r["program_name"] = name
            clean_rows.append(r)

        rows = clean_rows

        for r in rows:
            raw_program = r.get("program_name_raw") or r.get("program_name") or ""
            heuristic_name = clean_program_title(raw_program)

            # If the cleaned name is not actually a usable program label, skip row
            if not looks_like_program_name(heuristic_name):
                continue

            program_id = get_program_id(heuristic_name)

            checksum = md5_row(
                heuristic_name,
                r["fy"],
                r["amount"],
                r["page"],
            )

            bbox_val = r.get("bbox")
            bbox_json = json.dumps(bbox_val) if bbox_val else None

            # Default metadata (no AI)
            final_name = heuristic_name
            ai_name: str | None = None
            ai_brief: str | None = None
            name_source = "heuristic"

            # Decide whether this looks generic and should be refined by AI
            if needs_ai_refinement(heuristic_name):
                ai_name_tmp, ai_brief_tmp = await call_ai_for_program(
                    # use the raw program text as context for now
                    context=raw_program,
                    amount=r.get("amount"),
                    fy=r.get("fy"),
                    file_label=file_name,
                )
                if ai_name_tmp:
                    final_name = ai_name_tmp
                    ai_name = ai_name_tmp
                    ai_brief = ai_brief_tmp or None
                    name_source = "ai"

            await db.execute(
                """
                INSERT INTO rows (
                    table_id,
                    program_id,
                    program_name,
                    program_name_raw,
                    program_ai_name,
                    program_ai_brief,
                    program_name_source,
                    fy,
                    amount,
                    bbox,
                    page,
                    checksum
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                ON CONFLICT (table_id, checksum) DO UPDATE
                SET
                    program_id = EXCLUDED.program_id,
                    program_name = EXCLUDED.program_name,
                    program_name_raw = EXCLUDED.program_name_raw,
                    program_ai_name = EXCLUDED.program_ai_name,
                    program_ai_brief = EXCLUDED.program_ai_brief,
                    program_name_source = EXCLUDED.program_name_source,
                    fy = EXCLUDED.fy,
                    amount = EXCLUDED.amount,
                    bbox = EXCLUDED.bbox,
                    page = EXCLUDED.page
                """,
                table_id,
                program_id,
                final_name,                   # program_name (final)
                raw_program or None,          # program_name_raw
                ai_name,                      # program_ai_name
                ai_brief,                     # program_ai_brief
                name_source,                  # program_name_source
                r["fy"],
                r["amount"],
                bbox_json,
                r["page"],
                checksum,
            )

        audit = await audit_table(str(table_id), db)
        return {
            "table_id": str(table_id),
            "audit": audit,
            "count": len(rows),
        }

    except HTTPException:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
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
            SELECT id, program_name, fy, amount, page
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

class DiffIn(BaseModel):
    prev_table_id: str
    curr_table_id: str

@app.post("/diff")
async def create_diff(diff_in: DiffIn, db=Depends(get_db)):
    prev_id = diff_in.prev_table_id
    curr_id = diff_in.curr_table_id

    # Create a diff record
    diff_row = await db.fetchrow(
        """
        INSERT INTO diffs (prev_table_id, curr_table_id)
        VALUES ($1::uuid, $2::uuid)
        RETURNING id
        """,
        prev_id,
        curr_id,
    )
    diff_id = diff_row["id"]

    # Build per-program differences
    # MVP: join on program_name text
    await db.execute(
        """
        WITH prev AS (
        SELECT
            COALESCE(program_id, program_name) AS program_key,
            program_name,
            amount AS prev_amount
        FROM rows
        WHERE table_id = $1::uuid
        ),
        curr AS (
        SELECT
            COALESCE(program_id, program_name) AS program_key,
            program_name,
            amount AS curr_amount
        FROM rows
        WHERE table_id = $2::uuid
        ),
        joined AS (
        SELECT
            COALESCE(p.program_name, c.program_name) AS program_name,
            p.prev_amount,
            c.curr_amount
        FROM prev p
        FULL OUTER JOIN curr c
            ON p.program_key = c.program_key
        )
        INSERT INTO diff_rows (
        diff_id, program_name, prev_amount, curr_amount, delta_abs, delta_pct
        )
        SELECT
        $3::uuid,
        program_name,
        prev_amount,
        curr_amount,
        COALESCE(curr_amount, 0) - COALESCE(prev_amount, 0) AS delta_abs,
        CASE
            WHEN prev_amount IS NULL OR prev_amount = 0 THEN NULL
            ELSE ((COALESCE(curr_amount, 0) - prev_amount)::double precision / prev_amount) * 100.0
        END AS delta_pct
        FROM joined
        WHERE prev_amount IS NOT NULL OR curr_amount IS NOT NULL
        """,
        prev_id,
        curr_id,
        diff_id,
    )


    return {"diff_id": str(diff_id)}

@app.get("/diff/{diff_id}")
async def get_diff(diff_id: str, db=Depends(get_db)):
    rows = await db.fetch(
        """
        SELECT
          dr.program_name,
          dr.prev_amount,
          dr.curr_amount,
          dr.delta_abs,
          dr.delta_pct,
          rc.id AS curr_row_id
        FROM diff_rows dr
        JOIN diffs d
          ON dr.diff_id = d.id
        LEFT JOIN rows rc
          ON rc.table_id = d.curr_table_id
         AND rc.program_name = dr.program_name
        WHERE dr.diff_id = $1::uuid
        ORDER BY ABS(COALESCE(dr.delta_abs, 0)) DESC, dr.program_name

        """,
        diff_id,
    )

    return [
        {
            "program_name": r["program_name"],
            "prev_amount": r["prev_amount"],
            "curr_amount": r["curr_amount"],
            "delta_abs": r["delta_abs"],
            "delta_pct": r["delta_pct"],
            "curr_row_id": str(r["curr_row_id"]) if r["curr_row_id"] else None,
        }
        for r in rows
    ]

class RenderIn(BaseModel):
    row_id: str

import json
from decimal import Decimal
from fastapi import HTTPException

@app.post("/render")
async def render_cell(body: RenderIn, db=Depends(get_db)):
    row = await db.fetchrow(
        """
        SELECT r.page, r.bbox, f.stored_url
        FROM rows r
        JOIN tables t ON r.table_id = t.id
        JOIN files  f ON t.file_id = f.id
        WHERE r.id = $1::uuid
        """,
        body.row_id,
    )
    print(f"render ==== row {row}")
    if not row:
        raise HTTPException(status_code=404, detail="row not found")

    raw_bbox = row["bbox"]
    print(f"render ==== bbox raw {raw_bbox} ({type(raw_bbox)})")

    if not raw_bbox:
        raise HTTPException(status_code=400, detail="no bbox stored for this row")

    # --------- normalize bbox into [x0, top, x1, bottom] ---------
    bbox = None

    # Case 1: already list/tuple/array-like
    if isinstance(raw_bbox, (list, tuple)):
        try:
            bbox = [float(v) if not isinstance(v, Decimal) else float(v) for v in raw_bbox]
        except (TypeError, ValueError):
            bbox = None

    # Case 2: dict with keys
    elif isinstance(raw_bbox, dict):
        try:
            bbox = [
                float(raw_bbox["x0"]),
                float(raw_bbox["top"]),
                float(raw_bbox["x1"]),
                float(raw_bbox["bottom"]),
            ]
        except (KeyError, TypeError, ValueError):
            bbox = None

    # Case 3: string from DB (json or comma-separated or "(..,..,..,..)")
    elif isinstance(raw_bbox, str):
        s = raw_bbox.strip()
        if s:
            # try JSON first
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                # fallback: strip parentheses and split on commas
                s2 = s.replace("(", "").replace(")", "")
                parts = [p.strip() for p in s2.split(",")]
                if len(parts) == 4:
                    try:
                        bbox = [float(p) for p in parts]
                    except ValueError:
                        bbox = None
            else:
                # json was ok
                if isinstance(parsed, (list, tuple)):
                    try:
                        bbox = [float(v) for v in parsed]
                    except (TypeError, ValueError):
                        bbox = None
                elif isinstance(parsed, dict):
                    try:
                        bbox = [
                            float(parsed["x0"]),
                            float(parsed["top"]),
                            float(parsed["x1"]),
                            float(parsed["bottom"]),
                        ]
                    except (KeyError, TypeError, ValueError):
                        bbox = None

    if not bbox or len(bbox) < 4:
        print("render ==== normalized bbox is invalid:", bbox)
        raise HTTPException(status_code=400, detail="invalid bbox format for this row")

    x0, top, x1, bottom = bbox[:4]
    # --------- end bbox normalization ---------

    page_num = row["page"] or 1
    stored_url = row["stored_url"]

    pdf_path = download_file_to_tmp(stored_url)

    fd, png_path = tempfile.mkstemp(suffix=".png")
    _os.close(fd)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_index = max(0, page_num - 1)
            if page_index >= len(pdf.pages):
                page_index = len(pdf.pages) - 1
            page = pdf.pages[page_index]

            img = page.to_image(resolution=150)
            img.draw_rect((x0, top, x1, bottom), fill=None, stroke="red", stroke_width=3)
            img.save(png_path, format="PNG")
    finally:
        try:
            _os.remove(pdf_path)
        except:
            pass

    key = f"renders/{body.row_id}.png"
    s3.upload_file(png_path, BUCKET, key, ExtraArgs={"ContentType": "image/png"})
    try:
        _os.remove(png_path)
    except:
        pass

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=3600,
    )

    return {"url": url}

@app.get("/export/csv")
async def export_csv(diff_id: str, db=Depends(get_db)):
    # Query diff rows for the given diff_id, including program_id if available
    rows = await db.fetch(
        """
        SELECT 
          COALESCE(rp.program_id, rc.program_id) AS program_id,
          dr.program_name,
          dr.prev_amount,
          dr.curr_amount,
          dr.delta_abs,
          dr.delta_pct
        FROM diff_rows dr
        JOIN diffs d ON dr.diff_id = d.id
        LEFT JOIN rows rp 
          ON rp.table_id = d.prev_table_id AND rp.program_name = dr.program_name
        LEFT JOIN rows rc 
          ON rc.table_id = d.curr_table_id AND rc.program_name = dr.program_name
        WHERE dr.diff_id = $1::uuid
        """,
        diff_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Diff not found or no data")
    # Write CSV to an in-memory buffer
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["program_id", "name", "prev", "curr", "delta_abs", "delta_pct"])
    for r in rows:
        writer.writerow([
            r["program_id"] or "",       # use empty string if None
            r["program_name"],
            r["prev_amount"] if r["prev_amount"] is not None else "",
            r["curr_amount"] if r["curr_amount"] is not None else "",
            r["delta_abs"] if r["delta_abs"] is not None else "",
            r["delta_pct"] if r["delta_pct"] is not None else ""
        ])
    csv_data = output.getvalue()
    output.close()
    # Return CSV file response
    filename = f"diff_{diff_id}.csv"
    return Response(content=csv_data, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})

@app.get("/brief/pdf")
async def brief_pdf(diff_id: str, db=Depends(get_db)):
    # Query top 5 increases and top 5 cuts for the diff
    # (largest positive delta_abs and largest negative delta_abs)
    incs = await db.fetch(
        """
        SELECT dr.program_name, dr.prev_amount, dr.curr_amount, dr.delta_abs, dr.delta_pct,
               pf.name AS prev_file, rp.page AS prev_page, rp.checksum AS prev_checksum,
               cf.name AS curr_file, rc.page AS curr_page, rc.checksum AS curr_checksum
        FROM diff_rows dr
        JOIN diffs d ON dr.diff_id = d.id
        LEFT JOIN rows rp ON rp.table_id = d.prev_table_id AND rp.program_name = dr.program_name
        LEFT JOIN tables tp ON rp.table_id = tp.id
        LEFT JOIN files pf ON tp.file_id = pf.id
        LEFT JOIN rows rc ON rc.table_id = d.curr_table_id AND rc.program_name = dr.program_name
        LEFT JOIN tables tc ON rc.table_id = tc.id
        LEFT JOIN files cf ON tc.file_id = cf.id
        WHERE dr.diff_id = $1::uuid AND dr.delta_abs > 0
        ORDER BY dr.delta_abs DESC
        LIMIT 5;
        """,
        diff_id,
    )
    cuts = await db.fetch(
        """
        SELECT dr.program_name, dr.prev_amount, dr.curr_amount, dr.delta_abs, dr.delta_pct,
               pf.name AS prev_file, rp.page AS prev_page, rp.checksum AS prev_checksum,
               cf.name AS curr_file, rc.page AS curr_page, rc.checksum AS curr_checksum
        FROM diff_rows dr
        JOIN diffs d ON dr.diff_id = d.id
        LEFT JOIN rows rp ON rp.table_id = d.prev_table_id AND rp.program_name = dr.program_name
        LEFT JOIN tables tp ON rp.table_id = tp.id
        LEFT JOIN files pf ON tp.file_id = pf.id
        LEFT JOIN rows rc ON rc.table_id = d.curr_table_id AND rc.program_name = dr.program_name
        LEFT JOIN tables tc ON rc.table_id = tc.id
        LEFT JOIN files cf ON tc.file_id = cf.id
        WHERE dr.diff_id = $1::uuid AND dr.delta_abs < 0
        ORDER BY dr.delta_abs ASC
        LIMIT 5;
        """,
        diff_id,
    )
    if not incs and not cuts:
        raise HTTPException(status_code=404, detail="No differences found for this diff")
    # Set up PDF canvas
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter  # 612x792 points for letter size

    text_y = height - 50  # start 50 points from top
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, text_y, "Top Increases")
    text_y -= 20
    pdf.setFont("Helvetica", 11)
    if incs:
        for row in incs:
            # Example line: Program Name: prev -> curr ( +delta_abs, +delta_pct% )
            name = row["program_name"] or "Unnamed Program"
            prev = row["prev_amount"]
            curr = row["curr_amount"]
            delta_abs = row["delta_abs"]
            delta_pct = row["delta_pct"]
            # Format values for display (use comma separators for amounts, one decimal for pct)
            prev_str = f"{prev:,}" if prev is not None else "—"
            curr_str = f"{curr:,}" if curr is not None else "—"
            # delta_abs sign and formatting
            sign = "+" if delta_abs and delta_abs > 0 else ""
            delta_abs_str = f"{sign}{delta_abs:,}" if delta_abs is not None else "0"
            delta_pct_str = f"({sign}{delta_pct:.1f}%)" if delta_pct is not None else ""
            line = f"{name}: {prev_str} → {curr_str}   ({delta_abs_str} {delta_pct_str})"
            pdf.drawString(70, text_y, line)
            text_y -= 15
    else:
        pdf.drawString(70, text_y, "None")
        text_y -= 15

    # Top Cuts section
    text_y -= 10  # small gap before next section
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, text_y, "Top Cuts")
    text_y -= 20
    pdf.setFont("Helvetica", 11)
    if cuts:
        for row in cuts:
            name = row["program_name"] or "Unnamed Program"
            prev = row["prev_amount"]
            curr = row["curr_amount"]
            delta_abs = row["delta_abs"]
            delta_pct = row["delta_pct"]
            prev_str = f"{prev:,}" if prev is not None else "—"
            curr_str = f"{curr:,}" if curr is not None else "—"
            # For cuts, delta_abs will be negative
            sign = "-" if delta_abs and delta_abs < 0 else ""
            delta_abs_str = f"{delta_abs:,}" if delta_abs is not None else "0"
            delta_pct_str = f"({delta_pct:.1f}%)" if delta_pct is not None else ""
            line = f"{name}: {prev_str} → {curr_str}   ({delta_abs_str} {delta_pct_str})"
            pdf.drawString(70, text_y, line)
            text_y -= 15
    else:
        pdf.drawString(70, text_y, "None")
        text_y -= 15

    # Footer with sources
    text_y -= 10
    pdf.setFont("Helvetica-Oblique", 10)
    pdf.drawString(50, text_y, "Sources:")
    text_y -= 14
    pdf.setFont("Helvetica", 9)
    # List each program's sources (file name, page, checksum)
    for row in list(incs) + list(cuts):
        name = row["program_name"] or "Unnamed Program"
        # Previous source (if exists)
        if row["prev_amount"] is not None and row["prev_file"]:
            prev_src = f"Prev source for '{name}': {row['prev_file']} p.{row['prev_page']} (checksum {row['prev_checksum']})"
            pdf.drawString(60, text_y, prev_src)
            text_y -= 12
        # Current source (if exists)
        if row["curr_amount"] is not None and row["curr_file"]:
            curr_src = f"Curr source for '{name}': {row['curr_file']} p.{row['curr_page']} (checksum {row['curr_checksum']})"
            pdf.drawString(60, text_y, curr_src)
            text_y -= 12
    # Finalize PDF
    pdf.showPage()
    pdf.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    # Return PDF response
    filename = f"brief_{diff_id}.pdf"
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'})

@app.get("/tables/latest")
async def list_latest_tables(limit: int = 20, db=Depends(get_db)):
    rows = await db.fetch(
        """
        SELECT
          t.id   AS table_id,
          f.name AS file_name,
          t.label AS table_label,
          t.created_at,
          f.stored_url
        FROM tables t
        JOIN files f ON t.file_id = f.id
        ORDER BY t.created_at DESC
        LIMIT $1
        """,
        limit,
    )

    visible: list[dict[str, Any]] = []

    for r in rows:
        stored_url = r["stored_url"]
        try:
            bucket, key = s3_url_to_bucket_key(stored_url)
            # HEAD is cheap and doesn’t pull the whole object
            s3.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            # If the object is gone, skip this table
            if code in ("NoSuchKey", "404", "NotFound"):
                continue
            # any other S3 error: be conservative and still include it
        except Exception:
            # if S3 is flaky, don't blow up the endpoint
            pass

        visible.append({
            "table_id": str(r["table_id"]),
            "file_name": r["file_name"],
            "table_label": r["table_label"],
            "created_at": r["created_at"].isoformat(),
        })

    return visible

@app.get("/tables/{table_id}")
async def get_table(table_id: str, db=Depends(get_db)):
    # Table metadata
    table = await db.fetchrow(
        """
        SELECT t.id, t.label, t.created_at, f.name AS file_name
        FROM tables t
        JOIN files f ON t.file_id = f.id
        WHERE t.id = $1::uuid
        """,
        table_id,
    )

    if not table:
        raise HTTPException(status_code=404, detail="table not found")

    # Table rows
    rows = await db.fetch(
        """
        SELECT program_name, amount, page, bbox, fy
        FROM rows
        WHERE table_id = $1::uuid
        ORDER BY page, program_name
        """,
        table_id,
    )

    return {
        "table_id": str(table["id"]),
        "file_name": table["file_name"],
        "table_label": table["label"],
        "created_at": table["created_at"].isoformat(),
        "rows": [
            {
                "program_name": r["program_name"],
                "amount": r["amount"],
                "page": r["page"],
                "bbox": r["bbox"],
                "fy": r["fy"],
            }
            for r in rows
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
