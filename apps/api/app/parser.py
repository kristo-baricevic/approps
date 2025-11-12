import os, re, hashlib, tempfile
import pdfplumber
from urllib.parse import urlparse
from typing import Optional, Tuple, Any, List, Dict
from app.storage import s3
from botocore.exceptions import ClientError
from fastapi import HTTPException

NUM_RE = re.compile(r"\(?\$?\s*[\d,]+(?:\.\d{1,2})?\)?")

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

def parse_pdf_to_rows(local_path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with pdfplumber.open(local_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            # Try lines-first, then text flow as a fallback
            settings_candidates = [
                {"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 3},
                {"vertical_strategy": "text",  "horizontal_strategy": "text",  "text_x_tolerance": 2, "text_y_tolerance": 2},
            ]

            tables = []
            for ts in settings_candidates:
                try:
                    tables = page.extract_tables(table_settings=ts) or []
                except Exception:
                    tables = []
                if tables:
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
                    if not program:
                        continue

                    # pick the rightmost numeric cell as amount
                    amount_cell = None
                    for cell in reversed(row):
                        if cell and NUM_RE.search(str(cell)):
                            amount_cell = str(cell)
                            break
                    if not amount_cell:
                        continue

                    amount_val = clean_amount(amount_cell)
                    if amount_val is None:
                        continue

                    bbox = find_amount_bbox(page, amount_cell) or None

                    out.append({
                        "program_name": program,
                        "amount": amount_val,
                        "page": pidx,
                        "bbox": bbox,
                        "fy": fy,
                    })
    return out

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
