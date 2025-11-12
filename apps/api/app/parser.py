import os, re, hashlib, tempfile
import pdfplumber
from urllib.parse import urlparse
from typing import Optional, Tuple, Any, List, Dict
from app.storage import s3
from botocore.exceptions import ClientError
from fastapi import HTTPException

NUM_RE = re.compile(r"\(?\$?\s*[\d,]+(?:\.\d{1,2})?\)?")
FY_RE  = re.compile(r"\bFY\s*([12]\d{3})\b", re.I)

AMT_RE = re.compile(
    r"""
    (?P<prefix>\$)?\s*
    (?P<num>\d[\d,]*(?:\.\d+)?)
    \s*
    (?P<unit>billion|million|thousand|bn|m|k)?
    """,
    re.I | re.X,
)

NEG_CUES = re.compile(r"\b(cut|cuts|decrease|decreases|reduction|reduces|below|(-|\u2013) ?less)\b", re.I)
IS_YEAR = re.compile(r"\b(19|20)\d{2}\b")  # 1900–2099
IS_EO_ID = re.compile(r"\b14\d{2,}\b")     # e.g., 14173, 14321 etc.

def to_whole_dollars(num_str: str, unit: Optional[str], had_dollar_prefix: bool) -> Optional[int]:
    if not num_str or not re.search(r"\d", num_str):
        return None
    # If there’s no $ and no unit, and it looks like a year/EO, skip
    if not had_dollar_prefix and not unit:
        if IS_YEAR.fullmatch(num_str) or IS_EO_ID.fullmatch(num_str):
            return None
    try:
        x = float(num_str.replace(",", ""))
    except ValueError:
        return None
    u = (unit or "").lower()
    if u in ("billion", "bn"): mult = 1_000_000_000
    elif u in ("million", "m"): mult = 1_000_000
    elif u in ("thousand", "k"): mult = 1_000
    else: mult = 1
    # Heuristic: naked small integers (e.g., “1060”) are often awards; keep them.
    return int(round(x * mult))

def extract_program_phrase(line: str) -> str:
    """
    Grab a concise 'what' phrase after a funding verb up to a comma/that-clause.
    """
    s = line.strip()
    # common leads
    verbs = [
        r"providing\b", r"provide\b", r"provides\b", r"maintaining\b", r"maintains\b",
        r"increasing\b", r"increases\b", r"decreasing\b", r"decreases\b",
        r"reducing\b", r"reduces\b", r"eliminating\b", r"eliminates\b",
        r"a total discretionary allocation of\b", r"total discretionary allocation of\b"
    ]
    m = re.search(r"(?:" + "|".join(verbs) + r")(.+)", s, flags=re.I)
    if m:
        frag = m.group(1)
    else:
        # fallback: after amount
        m2 = re.search(AMT_RE, s)
        frag = s[m2.end():] if m2 else s

    # trim at strong stopper
    frag = re.split(r"[.;]|—|–", frag, maxsplit=1)[0]
    # strip leading preps/stop-words
    frag = re.sub(r"^\s*(for|to|of|in|by|at|on|toward|towards|which|that|the)\s+", "", frag, flags=re.I)
    frag = re.sub(r"\s+", " ", frag).strip()
    # keep it readable but short
    return frag[:160]


def s3_url_to_bucket_key(stored_url: str) -> Tuple[str, str]:
    # s3://<bucket>/<key>
    u = urlparse(stored_url)
    return (u.netloc, u.path.lstrip("/"))

def clean_amount(s: str) -> Optional[int]:
    if not s: return None
    t = s.strip()
    m = NUM_RE.search(t)
    if not m: return None
    t = m.group(0)
    neg = t.startswith("(") and t.endswith(")")
    t = t.replace("(", "").replace(")", "").replace("$", "").replace(",", "").strip()
    try:
        val = float(t)
    except ValueError:
        return None
    val = int(round(val))  # whole dollars MVP
    return -val if neg else val

def norm_header(h: str) -> str:
    if not h: return ""
    h = h.strip().lower()
    h = h.replace("\n"," ").replace("\r"," ")
    h = re.sub(r"\s+", " ", h)
    return h

def detect_fy_from_headers(headers: List[str]) -> Optional[int]:
    for h in headers:
        m = re.search(r"fy\s*([12]\d{3})", h, flags=re.I)
        if m: return int(m.group(1))
    return None

def md5_row(program_name: str, fy: Optional[int], amount: Optional[int], page: int) -> str:
    payload = f"{program_name}|{fy or ''}|{amount or ''}|{page}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def find_amount_bbox(page: pdfplumber.page.Page, amount_text: str) -> Optional[List[float]]:
    # try to locate by stripped variants (with/without $/,)
    cand = set([amount_text, amount_text.replace("$",""), amount_text.replace(",",""),
                amount_text.replace("$","").replace(",","")])
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    for w in words:
        txt = w.get("text","").strip()
        if txt in cand:
            return [w["x0"], w["top"], w["x1"], w["bottom"]]
    return None

import re
NUM_CURRENCY_RE = re.compile(r"\(?\$\s*[\d,]+(?:\.\d{1,2})?\)?")
UNIT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(billion|million)\b", re.I)

def parse_pdf_prose_amounts(local_path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with pdfplumber.open(local_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text:
                continue
            # split on bullet-like markers too
            parts = re.split(r"(?:\n|\r|\u2022|\u2023|•|◦|o\s)", text)
            for raw in parts:
                line = (raw or "").strip()
                if len(line) < 6:
                    continue

                # find the *rightmost* amount (often the main figure in these bullets)
                matches = list(AMT_RE.finditer(line))
                if not matches:
                    continue

                m = matches[-1]
                amt = to_whole_dollars(
                    m.group("num"),
                    m.group("unit"),
                    had_dollar_prefix=bool(m.group("prefix")),
                )
                if amt is None:
                    continue

                # polarity
                if NEG_CUES.search(line):
                    amt = -abs(amt)

                prog = extract_program_phrase(line)
                # basic quality gate: at least 2 alphabetic tokens (avoid "•", "o", "AGENCIES ACT, 2026", etc.)
                if len(re.findall(r"[A-Za-z]{2,}", prog)) < 2:
                    # If it’s the top-line “total discretionary allocation ...” keep it
                    if "total discretionary allocation" not in prog.lower():
                        continue

                out.append({
                    "program_name": prog,
                    "amount": amt,
                    "page": pidx,
                    "bbox": None,
                    "fy": None,
                })
    return out


def parse_pdf_to_rows(local_path: str) -> list[dict]:
    out = []
    with pdfplumber.open(local_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            # tables only; skip prose-only pages
            settings_candidates = [
                {"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 3},
                {"vertical_strategy": "text",  "horizontal_strategy": "text",  "text_x_tolerance": 2, "text_y_tolerance": 2},
            ]
            tables = []
            for ts in settings_candidates:
                try:
                    t = page.extract_tables(table_settings=ts) or []
                except Exception:
                    t = []
                if t:
                    tables = t
                    break
            if not tables:
                continue

            for tbl in tables:
                if not tbl or len(tbl) < 2:
                    continue
                headers = [norm_header(c or "") for c in tbl[0]]
                fy = detect_fy_from_headers(headers)

                for row in tbl[1:]:
                    if not row: 
                        continue
                    program = (row[0] or "").strip()
                    # filter junky program names
                    if len(program) < 4:
                        continue
                    if len([t for t in re.findall(r"[A-Za-z]+", program) if len(t) >= 2]) < 2:
                        continue

                    # pick rightmost money-looking cell
                    amount_cell = None
                    for cell in reversed(row):
                        if not cell:
                            continue
                        s = str(cell)
                        if NUM_CURRENCY_RE.search(s):
                            amount_cell = s
                            break
                        m = UNIT_RE.search(s)
                        if m:
                            amount_cell = s
                            break
                    if not amount_cell:
                        continue

                    # normalize amount (handles $x,xxx and N billion/million)
                    amt = clean_amount(amount_cell)
                    if amt is None:
                        # try unit conversion
                        m = UNIT_RE.search(amount_cell)
                        if m:
                            val = float(m.group(1))
                            unit = m.group(2).lower()
                            mult = 1_000_000_000 if unit == "billion" else 1_000_000
                            amt = int(round(val * mult))
                    if amt is None:
                        continue

                    bbox = find_amount_bbox(page, amount_cell) or None
                    out.append({"program_name": program, "amount": amt, "page": pidx, "bbox": bbox, "fy": fy})
    return out

def parse_pdf_to_rows_combined(local_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        rows.extend(parse_pdf_to_rows(local_path))  # your existing (tables) function
    except Exception:
        # don’t fail the whole parse if tables logic breaks
        pass
    try:
        rows.extend(parse_pdf_prose_amounts(local_path))
    except Exception:
        pass
    return rows


def download_file_to_tmp(stored_url: str) -> str:
    bucket, key = s3_url_to_bucket_key(stored_url)
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf"); os.close(fd)
    try:
        s3.download_file(bucket, key, tmp_path)
    except ClientError as e:
        try:
            os.remove(tmp_path)
        except:
            pass
        # Make it a clear 400 so the browser isn’t stuck with a vague 500
        raise HTTPException(status_code=400, detail=f"S3 object not found: s3://{bucket}/{key}")
    return tmp_path
