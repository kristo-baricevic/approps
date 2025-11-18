import os
import asyncio
import json
from typing import Any, Dict, List, Optional

import asyncpg
import pdfplumber
from openai import AsyncOpenAI

from parser import download_file_to_tmp

DB_DSN = os.getenv("DATABASE_URL")
AI_MODEL = os.getenv("APPROPS_AI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def build_amount_phrase(amount: int) -> str:
    x = abs(amount)
    sign = "-" if amount < 0 else ""
    if x >= 1_000_000_000:
        return f"{sign}${x / 1_000_000_000:.1f} billion"
    if x >= 1_000_000:
        return f"{sign}${x / 1_000_000:.1f} million"
    if x >= 1_000:
        return f"{sign}${x / 1_000:.1f} thousand"
    return f"{sign}${x}"


async def fetch_rows_needing_ai(conn: asyncpg.Connection, limit: int = 50) -> List[asyncpg.Record]:
    sql = """
    SELECT
      r.id,
      r.program_name,
      r.amount,
      r.page,
      r.bbox,
      r.program_ai_name,
      r.program_ai_brief,
      f.stored_url,
      f.name AS file_name
    FROM rows r
    JOIN tables t ON r.table_id = t.id
    JOIN files f ON t.file_id = f.id
    WHERE r.amount IS NOT NULL
      AND (r.program_ai_name IS NULL OR r.program_ai_name = '')
    ORDER BY f.id, r.page, r.id
    LIMIT $1
    """
    rows = await conn.fetch(sql, limit)
    return list(rows)


async def call_ai_for_program(row: asyncpg.Record, page_text: str) -> Optional[Dict[str, str]]:
    amount = int(row["amount"])
    amount_phrase = build_amount_phrase(amount)
    program_snippet = (row["program_name"] or "").strip()
    file_name = row["file_name"]
    page = int(row["page"])

    payload = {
        "file_name": file_name,
        "page": page,
        "amount_numeric": amount,
        "amount_text": amount_phrase,
        "program_snippet": program_snippet,
        "page_text": page_text,
    }

    system_prompt = (
        "You are an expert analyst of U.S. congressional appropriations bills. "
        "Given text from an explanatory statement and a specific dollar amount, "
        "identify the primary program or activity that this amount funds. "
        "Return a short, human readable program name and a one sentence description. "
        "Do not include dollar values or citations in the name. If you cannot identify a clear program, "
        'use the name "UNKNOWN" and a brief description explaining that it is unclear.'
    )

    user_prompt = (
        "Return a JSON object with exactly two keys: "
        '"program_name" and "program_brief". '
        "Input:\n\n"
        + json.dumps(payload, indent=2)
    )

    completion = await client.chat.completions.create(
        model=AI_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = completion.choices[0].message.content or ""
    content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"program_name": content, "program_brief": ""}

    name = str(data.get("program_name", "")).strip()
    brief = str(data.get("program_brief", "")).strip()

    if not name:
        return None

    if len(name) > 255:
        name = name[:252] + "..."

    return {"program_name": name, "program_brief": brief}


async def enrich_batch_once(limit: int = 50) -> bool:
    if not DB_DSN:
        raise RuntimeError("DATABASE_URL is not set")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    pool = await asyncpg.create_pool(dsn=DB_DSN, min_size=1, max_size=5)

    any_updated = False

    async with pool.acquire() as conn:
        rows = await fetch_rows_needing_ai(conn, limit=limit)
        if not rows:
            await pool.close()
            return False

        by_file: Dict[str, List[asyncpg.Record]] = {}
        for r in rows:
            stored_url = r["stored_url"]
            by_file.setdefault(stored_url, []).append(r)

        for stored_url, file_rows in by_file.items():
            tmp_path = download_file_to_tmp(stored_url)
            try:
                with pdfplumber.open(tmp_path) as pdf:
                    for r in file_rows:
                        page_idx = int(r["page"]) - 1
                        if page_idx < 0 or page_idx >= len(pdf.pages):
                            continue
                        page = pdf.pages[page_idx]
                        page_text = page.extract_text() or ""
                        if not page_text.strip():
                            continue

                        ai_result = await call_ai_for_program(r, page_text)
                        if not ai_result:
                            continue

                        row_id = r["id"]
                        ai_name = ai_result["program_name"]
                        ai_brief = ai_result["program_brief"]

                        await conn.execute(
                            """
                            UPDATE rows
                            SET program_ai_name = $2,
                                program_ai_brief = $3,
                                program_name_source = 'ai'
                            WHERE id = $1::uuid
                            """,
                            str(row_id),
                            ai_name,
                            ai_brief,
                        )
                        any_updated = True
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    await pool.close()
    return any_updated


async def main():
    batch_size = int(os.getenv("APPROPS_AI_BATCH_SIZE", "25"))
    while True:
        updated = await enrich_batch_once(limit=batch_size)
        if not updated:
            break


if __name__ == "__main__":
    asyncio.run(main())
