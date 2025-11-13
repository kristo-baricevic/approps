import os, re, hashlib, tempfile
import pdfplumber
from urllib.parse import urlparse
from typing import Optional, Tuple, Any, List, Dict
from app.storage import s3
from botocore.exceptions import ClientError
from fastapi import HTTPException
import camelot
import re

NUM_CURRENCY_RE = re.compile(r"\(?\$\s*[\d,]+(?:\.\d{1,2})?\)?")
UNIT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(billion|million)\b", re.I)
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

# --- Program ID aliasing / normalization ---

# MVP alias map: fill this out over time.
PROGRAM_ALIAS_MAP: dict[str, str] = {
    "centers for disease control and prevention": "HHS_CDC",
    "centers for disease control": "HHS_CDC",
    "cdc": "HHS_CDC",
    "cdc programs": "HHS_CDC",

    "national institutes of health": "HHS_NIH",
    "nih": "HHS_NIH",
    "nih research": "HHS_NIH",

    "health resources and services administration": "HHS_HRSA",
    "hrsa": "HHS_HRSA",

    "substance abuse and mental health services administration": "HHS_SAMHSA",
    "samhsa": "HHS_SAMHSA",

    "administration for children and families": "HHS_ACF",
    "acf": "HHS_ACF",

    "head start": "HHS_ACF_HeadStart",
    "head start program": "HHS_ACF_HeadStart",
    "headstart": "HHS_ACF_HeadStart",
    "early head start": "HHS_ACF_HeadStart",

    "low income home energy assistance program": "HHS_LIHEAP",
    "liheap": "HHS_LIHEAP",
    "home energy assistance": "HHS_LIHEAP",

    "community health centers": "HHS_CHC",
    "community health center program": "HHS_CHC",
    "federally qualified health centers": "HHS_CHC",

    "department of education": "ED_Department",
    "education department": "ED_Department",

    "title i grants to local educational agencies": "ED_TitleI_LEA",
    "title i grants to leas": "ED_TitleI_LEA",
    "title i": "ED_TitleI_LEA",

    "special education grants to states": "ED_IDEA_B",
    "idea part b grants to states": "ED_IDEA_B",
    "idea part b": "ED_IDEA_B",

    "federal pell grants": "ED_Pell",
    "pell grants": "ED_Pell",
    "pell grant program": "ED_Pell",

    "workforce innovation and opportunity act youth activities": "DOL_WIOA_Youth",
    "wioa youth activities": "DOL_WIOA_Youth",
    "youth training activities": "DOL_WIOA_Youth",

    "workforce innovation and opportunity act adult activities": "DOL_WIOA_Adult",
    "wioa adult activities": "DOL_WIOA_Adult",
    "adult training activities": "DOL_WIOA_Adult",

    "job corps": "DOL_JobCorps",
    "jobcorps": "DOL_JobCorps",

    "supplemental nutrition assistance program": "USDA_SNAP",
    "snap": "USDA_SNAP",
    "food stamp program": "USDA_SNAP",

    "special supplemental nutrition program for women infants and children": "USDA_WIC",
    "wic": "USDA_WIC",

    "temporary assistance for needy families": "HHS_TANF",
    "tanf": "HHS_TANF",

    "housing choice vouchers": "HUD_HCV",
    "housing choice voucher program": "HUD_HCV",
    "section 8 housing choice vouchers": "HUD_HCV",

    "public housing fund": "HUD_PublicHousing",
    "public housing capital fund": "HUD_PublicHousing",
    "public housing operating fund": "HUD_PublicHousing",
}

JUNK_PROGRAM_RE = re.compile(
    r"""
    ^\s*(
        total|
        subtotal|
        administrative\ expenses|
        administration|
        offsetting\ collections|
        rescission|
        rescissions|
        appropriation|
        appropriations|
        new\ budget\ authority|
        advance\ appropriation|
        limitation\ on\ obligations|
        obligations\ limitation|
        discretionary\ budget\ authority|
        mandatory\ budget\ authority|
        salary\ and\ expenses|
        salaries\ and\ expenses
    )\b
    """,
    re.I | re.X,
)


def looks_like_program_name(program: str) -> bool:
    if not program:
        return False
    s = program.strip()
    if len(s) < 4:
        return False

    tokens = [t for t in re.findall(r"[A-Za-z]+", s) if len(t) >= 3]
    if len(tokens) < 2:
        return False

    if JUNK_PROGRAM_RE.search(s):
        return False

    if s.endswith(":"):
        return False

    return True


def normalize_program_name_for_id(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    s = re.sub(r"&", "and", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_program_id(raw_name: str) -> str:
    """
    Normalize a program name and map to a stable ID.
    If we don't know it yet, just use the normalized name itself as the ID.
    """
    norm = normalize_program_name_for_id(raw_name)
    if not norm:
        return ""
    return PROGRAM_ALIAS_MAP.get(norm, norm)


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

def extract_tables_with_camelot(local_path: str, page_number: int) -> List[List[List[str]]]:
    """
    Use Camelot to extract tables from a single page.
    Returns a list of tables, where each table is a list of rows,
    and each row is a list of strings, matching the shape expected by parse_pdf_to_rows.
    """
    try:
        camelot_tables = camelot.read_pdf(
            local_path,
            pages=str(page_number),
            flavor="lattice",  # good first guess for gov/appropriations docs
        )
    except Exception:
        return []

    tables: List[List[List[str]]] = []
    for table in camelot_tables:
        df = table.df  # pandas DataFrame
        # Convert DataFrame to list-of-lists of strings
        tbl: List[List[str]] = []
        for row in df.itertuples(index=False):
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            tbl.append(cells)
        if tbl:
            tables.append(tbl)

    return tables

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
    m = NUM_RE.search(amount_text)
    target = m.group(0) if m else amount_text

    target_digits = re.sub(r"[^\d.]", "", target)
    if not target_digits:
        return None

    try:
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=True,
            use_text_flow=False,
        )
    except TypeError:
        words = page.extract_words(use_text_flow=False, keep_blank_chars=True)

    for w in words:
        txt = (w.get("text") or "").strip()
        if not txt:
            continue

        w_digits = re.sub(r"[^\d.]", "", txt)
        if not w_digits:
            continue

        if (
            w_digits == target_digits
            or target_digits in w_digits
            or w_digits in target_digits
        ):
            return [w["x0"], w["top"], w["x1"], w["bottom"]]

    return None

def parse_pdf_to_rows(local_path: str) -> list[dict]:
    out: list[dict] = []
    with pdfplumber.open(local_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
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
                camelot_tables = extract_tables_with_camelot(local_path, pidx)
                tables = camelot_tables

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

                    if not looks_like_program_name(program):
                        continue

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

                    amt = clean_amount(amount_cell)
                    if amt is None:
                        m = UNIT_RE.search(amount_cell)
                        if m:
                            val = float(m.group(1))
                            unit = m.group(2).lower()
                            mult = 1_000_000_000 if unit == "billion" else 1_000_000
                            amt = int(round(val * mult))
                    if amt is None:
                        continue

                    m_amt = AMT_RE.search(amount_cell)
                    amount_text = m_amt.group(0) if m_amt else amount_cell
                    bbox = find_amount_bbox(page, amount_text) or None

                    out.append({
                        "program_name": program,
                        "amount": amt,
                        "page": pidx,
                        "bbox": bbox,
                        "fy": fy,
                    })
    return out

import re
NUM_CURRENCY_RE = re.compile(r"\(?\$\s*[\d,]+(?:\.\d{1,2})?\)?")
UNIT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(billion|million)\b", re.I)


def is_likely_appropriation_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False

    if not (NUM_CURRENCY_RE.search(s) or UNIT_RE.search(s)):
        return False

    s = re.sub(r"^\s*\d+\s+", "", s)
    s = re.sub(r"^\s*\d+\.\s+", "", s)

    if re.search(r"\bFor\b", s, flags=re.I):
        return True
    if re.search(r"\bnecessary expenses\b", s, flags=re.I):
        return True
    if re.search(r"\bappropriat(?:ed|ion|ions)\b", s, flags=re.I):
        return True
    if re.search(r"\bto remain available until\b", s, flags=re.I):
        return True
    if re.search(r"\bgrants to\b", s, flags=re.I):
        return True

    return False


def parse_pdf_prose_amounts(local_path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with pdfplumber.open(local_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text:
                continue

            parts = re.split(r"(?:\n|\r|\u2022|\u2023|•|◦|o\s)", text)
            for raw in parts:
                line = (raw or "").strip()
                if len(line) < 6:
                    continue

                if not is_likely_appropriation_line(line):
                    continue

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

                if NEG_CUES.search(line):
                    amt = -abs(amt)

                prog = extract_program_phrase(line)
                if len(re.findall(r"[A-Za-z]{2,}", prog)) < 2:
                    if "total discretionary allocation" not in prog.lower():
                        continue

                amount_text = m.group(0)
                bbox = find_amount_bbox(page, amount_text)

                out.append({
                    "program_name": prog,
                    "amount": amt,
                    "page": pidx,
                    "bbox": bbox,
                    "fy": None,
                })
    return out

def parse_pdf_to_rows_combined(local_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        rows.extend(parse_pdf_to_rows(local_path))
    except Exception:
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

def clean_program_title(text: str) -> str:
    s = (text or "").strip()

    # strip leading bullets like "(1)", "(A)", "(i)"
    s = re.sub(r"^\s*[\(\[]\s*[0-9A-Za-z]+\s*[\)\]]\s*", "", s)

    # grab the last "for ..." clause if present
    matches = list(re.finditer(r"\bfor\s+(.+)", s, flags=re.I))
    if matches:
        frag = matches[-1].group(1)
    else:
        frag = s

    # drop leading "the"/"such"
    frag = re.sub(r"^\s*(the|such)\s+", "", frag, flags=re.I)

    # cut at the first comma to avoid "..., of which ..." etc.
    frag = frag.split(",", 1)[0]

    # drop trailing junk like "that / which / shall / including"
    for pat in [r"\s+that\b", r"\s+which\b", r"\s+who\b", r"\s+shall\b", r"\s+including\b"]:
        frag = re.split(pat, frag, 1, flags=re.I)[0]

    frag = frag.strip(" ,.;:-")

    return frag.lower()
