import os, re, hashlib, tempfile
import pdfplumber
from urllib.parse import urlparse
from typing import Optional, Tuple, Any, List, Dict
from app.storage import s3
from botocore.exceptions import ClientError
from fastapi import HTTPException
import camelot
import re
import textwrap
from openai import OpenAI
import os

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment when creating OpenAI client")
    return OpenAI(api_key=api_key)



NUM_CURRENCY_RE = re.compile(
    r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\s*(?:million|billion|thousand)\b",
    re.I,
)
UNIT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(billion|million)\b", re.I)
NUM_RE = re.compile(r"\(?\$?\s*[\d,]+(?:\.\d{1,2})?\)?")
FY_RE  = re.compile(r"\bFY\s*([12]\d{3})\b", re.I)
OPENAI_MODEL_PROGRAMS = os.getenv("OPENAI_MODEL_PROGRAMS", "gpt-4.1-mini")

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


GENERIC_NAMES = {"unknown activity", "unknown program"}


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

SUBALLOCATION_PREFIXES = [
    "within the total",
    "within the amount",
    "within the funds",
    "within the amount provided",
    "of the funds provided",
    "of this amount",
    "of which",
    "the agreement includes",
    "agreement includes",
    "the agreement continues",
    "the agreement provides",
    "this activity",
    "this program",
    "these funds",
]

import asyncio

async def fill_missing_program_briefs(rows: list[dict]) -> list[dict]:
    """
    For every row that has a program name and amount but no program_ai_brief,
    call ai_label_program and fill in program_ai_name / program_ai_brief.

    Does not overwrite program_name. Your cleaned heuristic title stays as the
    display name; AI is only used for metadata.
    """
    tasks = []
    indices = []

    for i, row in enumerate(rows):
        # Skip rows with no money or obviously junk names
        amount = row.get("amount")
        name = row.get("program_name") or row.get("program_name_raw")
        if not amount or not name:
            continue

        # Already has a brief, nothing to do
        if row.get("program_ai_brief"):
            continue

        raw_snippet = row.get("program_name_raw") or row.get("program_name") or ""
        context = row.get("context") or ""

        tasks.append(ai_label_program(raw_snippet, context, amount))
        indices.append(i)

    if not tasks:
        return rows

    results = await asyncio.gather(*tasks)

    for idx, (ai_name, ai_brief) in zip(indices, results):
        row = rows[idx]

        # Only populate ai fields; do not touch the cleaned program_name
        if ai_name and not row.get("program_ai_name"):
            row["program_ai_name"] = ai_name
        if ai_brief:
            row["program_ai_brief"] = ai_brief

    return rows

async def parse_pdf_to_rows_with_ai(local_path: str) -> list[dict]:
    """
    Full pipeline used by the API:
    - parse tables and prose
    - postprocess and merge
    - fill in missing AI metadata (program_ai_name, program_ai_brief)
    """
    rows = parse_pdf_to_rows_combined(local_path)
    rows = await fill_missing_program_briefs(rows)
    return rows



def extract_amounts_from_text(text: str) -> list[int]:
    """
    Find all currency amounts in a chunk of text and return them as integers.
    Uses existing NUM_CURRENCY_RE and NUM_RE.
    """
    amounts: list[int] = []
    if not text:
        return amounts

    for m in NUM_CURRENCY_RE.finditer(text):
        token = m.group(0)

        # Pull out the numeric part
        num_match = NUM_RE.search(token)
        if not num_match:
            continue

        num_str = num_match.group(0)
        # Strip everything that is not a digit or dot
        num_str = re.sub(r"[^\d.]", "", num_str)
        if not num_str:
            continue

        value = float(num_str)

        # If the token has 'million' or 'billion', scale accordingly
        unit_match = UNIT_RE.search(token)
        if unit_match:
            unit_value = float(unit_match.group(1))
            unit = unit_match.group(2).lower()
            if unit.startswith("billion"):
                value = unit_value * 1_000_000_000
            elif unit.startswith("million"):
                value = unit_value * 1_000_000

        amounts.append(int(round(value)))

    return amounts


def parse_bbox_string(bbox_str: str):
    if not bbox_str:
        return None
    try:
        parts = [float(p.strip()) for p in bbox_str.strip("[]").split(",")]
        if len(parts) != 4:
            return None
        return parts
    except Exception:
        return None


def is_suballocation_text(text: str) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    for p in SUBALLOCATION_PREFIXES:
        if t.startswith(p):
            return True
    if t.startswith("within the"):
        return True
    return False


def is_total_text(text: str) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    if t.startswith("total "):
        return True
    if "total appropriation" in t or "total appropriations" in t:
        return True
    return False


def build_program_prompt(raw_snippet: str, context: Optional[str], amount: Optional[int]) -> str:
    amount_str = f"${amount:,}" if amount is not None else "an unspecified dollar amount"
    ctx = (context or "").strip()
    return textwrap.dedent(
        f"""
        You are helping label line items from a U.S. congressional appropriations explanatory statement.

        Task:
        1. Identify the specific program or activity funded by this dollar amount.
        2. Return:
           - a short human friendly program name
           - one sentence explaining what the program does in plain language

        Requirements:
        - The explanation sentence must not begin with any label such as 'Brief:', 'Summary:', or similar.
        - Do not add any prefixes of any kind.
        - Do not include dollar values in the name.
        - Do not reference sections, page numbers, or citations.
        - If the program cannot be determined, return "UNKNOWN" as the name and an empty explanation.

        Dollar amount: {amount_str}

        Raw snippet:
        {raw_snippet}

        Surrounding context:
        {ctx}
        """
    ).strip()

async def ai_label_program(
    raw_snippet: str,
    context: Optional[str],
    amount: Optional[int],
) -> tuple[str, str]:

    client = get_openai_client()
    prompt = build_program_prompt(raw_snippet, context, amount)

    resp = await asyncio.to_thread(
        client.chat.completions.create,
        model=OPENAI_MODEL_PROGRAMS,
        messages=[
            {"role": "system", "content": "You label budget line items from U.S. appropriations bills."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=128,
    )

    text = resp.choices[0].message.content.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Extract name and brief
    if not lines:
        name = "UNKNOWN"
        brief = ""
    elif len(lines) == 1:
        name = lines[0]
        brief = ""
    else:
        name = lines[0]
        brief = " ".join(lines[1:])

    # Cleanup
    if ":" in name:
        name = name.split(":", 1)[-1].strip()
    if name.upper().startswith("PROGRAM NAME"):
        name = name.split(":", 1)[-1].strip() or name
    name = name.strip() or "UNKNOWN"

    # If brief missing, enforce generation
    if not brief:
        brief_prompt = textwrap.dedent(
            f"""
            Write one plain language sentence describing what the following program does.
            Program: {name}
            Do not mention money, citations, or page numbers.
            """
        ).strip()

        resp2 = await asyncio.to_thread(
            client.chat.completions.create,
            model=OPENAI_MODEL_PROGRAMS,
            messages=[
                {"role": "system", "content": "You describe government programs."},
                {"role": "user", "content": brief_prompt},
            ],
            temperature=0.2,
            max_tokens=64,
        )

        brief = resp2.choices[0].message.content.strip().splitlines()[0].strip()

    return name, brief


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
    s = line.strip()

    if re.search(r"\.{5,}", s):
        before_dots = re.split(r"\.{3,}", s)[0]
        before_dots = before_dots.strip(" ,.;:-")
        if before_dots:
            return before_dots

    verbs = [
        r"providing\b", r"provide\b", r"provides\b", r"maintaining\b", r"maintains\b",
        r"increasing\b", r"increases\b", r"decreasing\b", r"decreases\b",
        r"reducing\b", r"reduces\b", r"eliminating\b", r"eliminates\b",
        r"a total discretionary allocation of\b", r"total discretionary allocation of\b",
    ]
    m = re.search(r"(?:" + "|".join(verbs) + r")(.+)", s, flags=re.I)
    if m:
        frag = m.group(1)
    else:
        m2 = re.search(AMT_RE, s)
        if m2:
            frag = s[m2.end():]
        else:
            frag = s

    frag = re.sub(r"^\s*(for|to)\s+", "", frag, flags=re.I)
    frag = re.split(r"[.;]", frag, 1)[0]
    frag = frag.strip(" ,.;:-")
    return frag



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
    import pdfplumber
    import re

    rows: list[dict] = []

    AMOUNT_RE = re.compile(r"\$?\s?(\d[\d,]{2,})\b")

    def extract_amount(text: str) -> int | None:
        m = AMOUNT_RE.findall(text)
        if not m:
            return None
        try:
            return int(m[-1].replace(",", ""))
        except:
            return None

    def line_center(y0, y1):
        return (y0 + y1) / 2

    with pdfplumber.open(local_path) as pdf:

        for page_index, page in enumerate(pdf.pages, start=1):
            word_objects = page.extract_words(
                keep_blank_chars=False,
                use_text_flow=True,
            )

            # build visual rows
            visual_rows: list[list[dict]] = []
            for w in word_objects:
                cy = line_center(w["top"], w["bottom"])
                placed = False
                for vr in visual_rows:
                    first = vr[0]
                    cy0 = line_center(first["top"], first["bottom"])
                    if abs(cy - cy0) <= 3:
                        vr.append(w)
                        placed = True
                        break
                if not placed:
                    visual_rows.append([w])

            # sort rows by vertical position
            visual_rows.sort(key=lambda r: line_center(r[0]["top"], r[0]["bottom"]))

            # rebuild lines with x ordering
            processed = []
            for vr in visual_rows:
                vr.sort(key=lambda w: w["x0"])
                text = " ".join(w["text"] for w in vr)
                x0 = min(w["x0"] for w in vr)
                top = min(w["top"] for w in vr)
                x1 = max(w["x1"] for w in vr)
                bottom = max(w["bottom"] for w in vr)

                # compute indent level relative to left margin
                indent = int(x0 // 20)

                processed.append(
                    {
                        "text": text.strip(),
                        "x0": x0,
                        "x1": x1,
                        "top": top,
                        "bottom": bottom,
                        "indent": indent,
                    }
                )

            # collapse hierarchy blocks
            blocks = []
            current_block = None

            for ln in processed:
                txt = ln["text"].strip()

                if ln["indent"] == 0:
                    # starting a new parent block
                    if current_block:
                        blocks.append(current_block)
                    current_block = {
                        "page": page_index,
                        "lines": [ln],
                    }
                else:
                    # attach to existing block if exists
                    if current_block:
                        t = ln["text"].strip().lower()

                        # NEW skip rule: prevent corrupting the program title
                        if re.match(r"^(less than|within|provided|the agreement|agreement)", t):
                            continue

                        current_block["lines"].append(ln)

                    else:
                        # no parent above, treat as parent anyway
                        current_block = {
                            "page": page_index,
                            "lines": [ln],
                        }

            if current_block:
                blocks.append(current_block)

            # finalize blocks into rows
            for block in blocks:
                page_num = block["page"]
                lines = block["lines"]

                full_text = " ".join(ln["text"] for ln in lines).strip()
                amount = extract_amount(full_text)

                # name = all text before the last amount occurrence
                name_text = full_text
                if amount is not None:
                    amt_token = f"{amount:,}"
                    idx = full_text.rfind(amt_token)
                    if idx > 0:
                        name_text = full_text[:idx].strip()

                # block bbox
                x0 = min(ln["x0"] for ln in lines)
                top = min(ln["top"] for ln in lines)
                x1 = max(ln["x1"] for ln in lines)
                bottom = max(ln["bottom"] for ln in lines)

                rows.append(
                    {
                        "program_name": name_text,
                        "program_name_raw": name_text,
                        "page": page_num,
                        "amount": amount,
                        "bbox": [float(x0), float(top), float(x1), float(bottom)],
                        "fy": None,
                    }
                )

    return rows


import re
NUM_CURRENCY_RE = re.compile(r"\(?\$\s*[\d,]+(?:\.\d{1,2})?\)?")
UNIT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(billion|million)\b", re.I)


def is_likely_appropriation_line(line: str) -> bool:
    s = (line or "").strip()
    print(f"is_likely_appropriation_line{s}")
    if not s:
        return False

    # NEW: reject boilerplate continuation lines that are not program titles
    low_signal_prefixes = (
        "agreement provides",
        "the agreement provides",
        "provided",
        "within",
        "the agreement continues",
        "continues",
        "agreement includes",
    )
    if s.lower().startswith(low_signal_prefixes):
        return False

    # Accept any line that clearly mentions a dollar or X million/billion.
    return bool(NUM_CURRENCY_RE.search(s) or UNIT_RE.search(s))


def looks_like_heading(line: str) -> bool:
    s = (line or "").strip()
    if len(s) < 4 or len(s) > 160:
        return False

    if NUM_CURRENCY_RE.search(s) or UNIT_RE.search(s):
        return False
    if re.search(r"\d", s):
        return False

    s_no_trail = s.rstrip(" .;:")
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", s_no_trail)
    if not words:
        return False
    if len(words) == 1 and len(words[0]) < 6:
        return False

    return True


def parse_pdf_prose_amounts(local_path: str) -> list[dict]:
    out: list[dict] = []
    with pdfplumber.open(local_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text:
                continue

            parts = re.split(r"(?:\n|\r|\u2022|\u2023|•|◦|o\s)", text)
            heading: Optional[str] = None

            for raw in parts:
                line = (raw or "").strip()
                if len(line) < 6:
                    continue

                if looks_like_heading(line):
                    heading = line
                    print(f"[prose] heading on page {pidx}: {heading}")
                    continue

                matches = list(AMT_RE.finditer(line))
                if not matches:
                    continue

                # keep your old heuristic, but only as an extra guard when there is no heading
                if not heading and not is_likely_appropriation_line(line):
                    print(f"[prose-skip] amt line with no heading and not appropriation-like on page {pidx}: {line}")
                    continue

                m = matches[-1]
                amt = to_whole_dollars(
                    m.group("num"),
                    m.group("unit"),
                    bool(m.group("prefix")),
                )
                if amt is None:
                    print(f"[prose-skip] could not parse amount on page {pidx}: {line}")
                    continue

                if NEG_CUES.search(line):
                    amt = -abs(amt)

                prog_from_line = extract_program_phrase(line)

                is_dotted_line = bool(re.search(r"\.{5,}", line))

                if heading and not is_suballocation_text(line) and not is_dotted_line:
                    prog = heading
                else:
                    prog = prog_from_line

                # if prog is still weak but we have a heading, fall back to heading
                if len(re.findall(r"[A-Za-z]{2,}", prog)) < 2:
                    if heading and prog != heading:
                        print(
                            f"[prose-fix] replacing weak prog {prog!r} with heading {heading!r} "
                            f"on page {pidx} from line: {line!r}"
                        )
                        prog = heading
                    elif "total discretionary allocation" not in prog.lower():
                        print(f"[prose-skip] weak program title on page {pidx}: {prog!r} from line: {line}")
                        continue

                amount_text = m.group(0)
                bbox = find_amount_bbox(page, amount_text)

                if heading:
                    context = f"{heading}: {line}"
                else:
                    context = line

                row = {
                    "program_name": prog,
                    "program_name_raw": line,
                    "program_name_source": "heuristic",
                    "amount": amt,
                    "page": pidx,
                    "bbox": bbox,
                    "fy": None,
                    "context": context,
                }
                print(f"[prose-row] page {pidx} amount={amt} prog={prog!r} raw={line!r}")
                out.append(row)
    return out

def parse_pdf_to_rows_combined(local_path: str) -> list[dict]:
    """
    Structured rows first.
    Then prose rows, but only on pages where structured rows are missing or weak.
    """
    structured = parse_pdf_to_rows(local_path)

    # Group structured rows by page
    structured_by_page: dict[int, list[dict]] = {}
    for r in structured:
        p = r.get("page")
        if not p:
            continue
        structured_by_page.setdefault(p, []).append(r)

    prose = parse_pdf_prose_amounts(local_path)

    filtered_prose: list[dict] = []

    for pr in prose:
        p = pr.get("page")
        if not p:
            continue

        page_structured = structured_by_page.get(p, [])

        # Per-page quality check
        has_good_structured = False
        if page_structured:
            # good criteria: ≥2 structured rows with normal-length titles
            good_titles = [
                r for r in page_structured
                if isinstance(r.get("program_name"), str)
                and len(r["program_name"].split()) >= 2
                and r.get("amount") is not None
            ]
            if len(good_titles) >= 2:
                has_good_structured = True

        # If structured rows are good, SKIP prose rows for this page
        if has_good_structured:
            continue

        # Otherwise, keep prose rows as fallback
        filtered_prose.append(pr)

    # Combine and postprocess
    all_rows = structured + filtered_prose
    all_rows = postprocess_rows(all_rows)
    return all_rows


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
    print(f"clean_program_title {s}")

    # Drop trailing "the agreement ..." clauses like:
    # "Delta States ... Grant Program.-The agreement continues ..."
    parts = re.split(r"\bthe agreement\b", s, 1, flags=re.I)
    if parts:
        s = parts[0]

    # Strip leading bullets like "(1)", "(A)", "(i)"
    s = re.sub(r"^\s*[\(\[]\s*[0-9A-Za-z]+\s*[\)\]]\s*", "", s)

    # Grab the last "for ..." clause if present (keeps "for X program")
    matches = list(re.finditer(r"\bfor\s+(.+)", s, flags=re.I))
    if matches:
        frag = matches[-1].group(1)
    else:
        frag = s

    # Drop leading "the"/"such"
    frag = re.sub(r"^\s*(the|such)\s+", "", frag, flags=re.I)

    # Cut at first comma to avoid "..., of which ..." tails
    frag = frag.split(",", 1)[0]

    # Drop trailing junk like "that / which / shall / including"
    for pat in [r"\s+that\b", r"\s+which\b", r"\s+who\b", r"\s+shall\b", r"\s+including\b"]:
        frag = re.split(pat, frag, 1, flags=re.I)[0]

    frag = frag.strip(" ,.;:-")
    print(f"clean_program_title frag {frag}")

    # Smart title-casing with acronym handling
    SMALL = {
        "and", "or", "for", "in", "of", "the", "a", "an",
        "to", "on", "at", "by", "with", "from", "into",
        "over", "per",
    }
    ACRONYMS = {
        "HIV", "AIDS", "SCID", "USAID", "NIH", "CDC", "SAMHSA",
        "HRSA", "HUD", "WIC", "SNAP", "TANF", "FBI", "DOJ",
        "HHS", "DOL", "SSA", "VA", "EPA", "USDA", "DOT",
    }

    tokens = re.split(r"(\s+)", frag)
    out: list[str] = []

    for i, tok in enumerate(tokens):
        if not tok:
            continue
        if tok.isspace():
            out.append(tok)
            continue

        lower = tok.lower()

        # Lowercase "small words" when not first token
        if i != 0 and lower in SMALL:
            out.append(lower)
            continue

        # Acronyms in parentheses, for example "(SCID)"
        m_paren = re.fullmatch(r"\(([A-Za-z0-9/&-]+)\)", tok)
        if m_paren and m_paren.group(1).isupper():
            inner = m_paren.group(1)
            if inner.upper() in ACRONYMS or len(inner) <= 4:
                out.append("(" + inner.upper() + ")")
                continue

        # Standalone acronyms, for example HIV, AIDS, NIH
        if tok.upper() in ACRONYMS:
            out.append(tok.upper())
            continue

        # Short all-caps tokens or ones with digits or slashes
        if re.fullmatch(r"[A-Z0-9/&-]+", tok):
            if len(tok) <= 3 or any(c.isdigit() or c in "/&" for c in tok):
                out.append(tok)
                continue

        # Default: normal title-case
        out.append(lower[:1].upper() + lower[1:])

    titled = "".join(out).strip()
    return titled



def _parse_bbox_for_row(bbox_str: str | None) -> Tuple[float, float, float, float] | None:
    if not bbox_str:
        return None
    try:
        nums = [float(x) for x in bbox_str.strip("[]").split(",")]
        if len(nums) != 4:
            return None
        return nums[0], nums[1], nums[2], nums[3]
    except Exception:
        return None

def _is_generic_name(name: str | None) -> bool:
    if not name:
        return True
    return name.strip().lower() in GENERIC_NAMES

def postprocess_rows(rows: list[dict]) -> list[dict]:
    filtered_rows: list[dict] = []
    for row in rows:
        ctx_raw = (row.get("context") or row.get("program_name_raw") or "")
        ctx = ctx_raw.lower()

        # Fix lines like "not less than $29,000,000, an increase of $2,000,000"
        if "not less than" in ctx and "increase of" in ctx:
            amounts_in_ctx = extract_amounts_from_text(ctx_raw)
            if amounts_in_ctx:
                new_amt = max(amounts_in_ctx)
                old_amt = row.get("amount") or 0
                if new_amt != old_amt:
                    print(
                        f"[postprocess] override not-less-than+increase "
                        f"amount {old_amt} -> {new_amt}"
                    )
                row["amount"] = new_amt

        amt = row.get("amount") or 0
        cite_hit = "p.l." in ctx or "u.s.c." in ctx or "section " in ctx

        if amt < 1_000 and cite_hit:
            print(f"[postprocess] drop cite row amount={amt} ctx={ctx[:120]}")
            continue

        if cite_hit and amt < 1_000_000:
            print(f"[postprocess] drop usc/p.l. row amount={amt} ctx={ctx[:120]}")
            continue

        # ... keep the rest of your existing postprocess_rows logic unchanged ...


        if amt < 1_000 and "days" in ctx and ("enactment" in ctx or "briefing" in ctx):
            print(f"[postprocess] drop days row amount={amt} ctx={ctx[:120]}")
            continue

        filtered_rows.append(row)

    rows = filtered_rows

    enriched: list[dict] = []
    for idx, row in enumerate(rows):
        bbox_field = row.get("bbox")
        bbox_vals = None
        if isinstance(bbox_field, (list, tuple)) and len(bbox_field) == 4:
            bbox_vals = tuple(float(x) for x in bbox_field)
        elif isinstance(bbox_field, str):
            bbox_vals = parse_bbox_string(bbox_field)

        if bbox_vals:
            left, top, right, bottom = bbox_vals
        else:
            left, top = 0.0, 0.0

        r = dict(row)
        r["_norm_bbox"] = bbox_vals
        r["_sort_key"] = (r.get("page") or 0, top, left, idx)
        print(
            f"[postprocess] keep row page={r.get('page')} "
            f"raw_bbox={bbox_field} norm_bbox={bbox_vals} "
            f"name={r.get('program_name')} amount={r.get('amount')}"
        )
        enriched.append(r)

    groups: dict[tuple, dict] = {}
    for r in enriched:
        page = r.get("page")
        if r["_norm_bbox"]:
            key = (page, tuple(r["_norm_bbox"]))
        else:
            key = (page, r.get("program_name"), r.get("amount"))

        if key not in groups:
            groups[key] = {
                "rows": [],
                "best_sort": r["_sort_key"],
            }
        groups[key]["rows"].append(r)
        if len(groups[key]["rows"]) > 1:
            names = [x.get("program_name") for x in groups[key]["rows"]]
            amts = [x.get("amount") for x in groups[key]["rows"]]
            print(
                f"[postprocess] group key={key} size={len(groups[key]['rows'])} "
                f"names={names} amounts={amts}"
            )
        if r["_sort_key"] < groups[key]["best_sort"]:
            groups[key]["best_sort"] = r["_sort_key"]

    merged: list[dict] = []
    for key, g in groups.items():
        rows_here = g["rows"]
        if len(rows_here) > 1:
            names = [x.get("program_name") for x in rows_here]
            amts = [x.get("amount") for x in rows_here]
            print(
                f"[postprocess] merging group key={key} "
                f"names={names} amounts={amts}"
            )

        base = dict(rows_here[0])

        best_name = base.get("program_name")
        best_raw = base.get("program_name_raw")
        best_source = base.get("program_name_source")

        for r in rows_here[1:]:
            name = r.get("program_name")
            source = r.get("program_name_source")
            raw = r.get("program_name_raw")

            if source == "heuristic" and name and "unknown" not in name.lower():
                best_name = name
                best_raw = raw
                best_source = source
                continue

            if source == "ai" and name:
                if not best_name or "unknown" in best_name.lower():
                    best_name = name
                    best_raw = raw
                    best_source = source

        base["program_name"] = best_name
        base["program_name_raw"] = best_raw
        base["program_name_source"] = best_source

        # Normalize the display title so it matches the program name
        # and has proper capitalization
        if base.get("program_name"):
            cleaned = clean_program_title(str(base["program_name"]))
            if cleaned:
                base["program_name"] = cleaned


        best_amount = base.get("amount") or 0
        for r in rows_here[1:]:
            amt = r.get("amount") or 0
            if amt > best_amount:
                best_amount = amt
        base["amount"] = best_amount

        raw_text = (best_raw or "").lower()
        if any(
            p in raw_text
            for p in [
                "within the total",
                "of which",
                "no less than",
            ]
        ):
            base["is_suballocation"] = True
        else:
            base["is_suballocation"] = False

        base["_sort_key"] = g["best_sort"]

        print(
            f"[postprocess] merged row page={base.get('page')} "
            f"key={key} name={base.get('program_name')} amount={base.get('amount')}"
        )

        merged.append(base)

    merged.sort(key=lambda r: r["_sort_key"])

    for r in merged:
        r.pop("_sort_key", None)
        r.pop("_norm_bbox", None)

    return merged

def choose_canonical(candidates: list[dict]) -> dict:
    good_ai = [
        r
        for r in candidates
        if r.get("program_ai_name")
        and not str(r.get("program_ai_name")).lower().startswith("unknown")
    ]
    if good_ai:
        base = dict(
            sorted(
                good_ai,
                key=lambda r: len(str(r.get("program_ai_name") or "")),
            )[-1]
        )
        base["program_name"] = base.get("program_ai_name")
        base["program_name_source"] = "ai"
        return base

    heur = [r for r in candidates if r.get("program_name_source") == "heuristic"]
    if heur:
        base = dict(
            sorted(
                heur,
                key=lambda r: len(str(r.get("program_name") or "")),
            )[-1]
        )
        return base

    return dict(candidates[0])

    ordered_groups = sorted(groups.values(), key=lambda g: g["sort_key"])
    final_rows: list[dict] = []
    for g in ordered_groups:
        chosen = choose_canonical(g["rows"])
        final_rows.append(chosen)

    current_parent_idx: int | None = None
    for idx, row in enumerate(final_rows):
        raw = row.get("program_name_raw") or ""
        name = row.get("program_name") or ""
        combined = " ".join(
            x for x in [row.get("program_name"), row.get("program_ai_name"), raw] if x
        )

        row["is_suballocation"] = is_suballocation_text(raw)
        row["is_total"] = is_total_text(combined)

        if row["is_suballocation"]:
            if current_parent_idx is not None:
                row["parent_row_id"] = final_rows[current_parent_idx].get("row_id")
        else:
            current_parent_idx = idx

        row["display_order"] = idx

    parent_ids = {
        r["parent_row_id"] for r in final_rows if r.get("parent_row_id") is not None
    }
    for r in final_rows:
        r["has_suballocations"] = r.get("row_id") in parent_ids

    for r in final_rows:
        if "_sort_key" in r:
            del r["_sort_key"]

    return final_rows
