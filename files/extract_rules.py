"""Baseline extractor: PDF text (pdfplumber) or OCR (Tesseract) + hand-written rules.

This is how many AP teams automate today: a template per supplier layout. The
rules below were written for layouts A-D only. Layout E is a supplier format
the rules have never seen, which is what happens every time a new vendor is
onboarded.

Run:  python src/extract_rules.py
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image
from rapidfuzz import fuzz, process

from common import INVOICE_DIR, OUTPUT

FIELDS = ["vendor_tax_id", "vendor_name", "invoice_number", "invoice_date", "due_date", "po_number",
          "currency", "subtotal", "tax", "total", "terms_days", "is_credit_memo"]
CODES = ["USD", "CAD", "EUR", "GBP", "SGD", "INR"]
SYMBOLS = {"C$": "CAD", "S$": "SGD", "£": "GBP", "€": "EUR", "Rs.": "INR", "$": "USD"}


# ---------- reading the page ----------
def rows_from_words(words: pd.DataFrame) -> str:
    """Rebuild reading order row by row (labels and values on the same line),
    instead of Tesseract's block order, which splits label and value columns."""
    words = words.assign(mid=words["top"] + words["height"] / 2).sort_values("mid")
    line_h = float(words["height"].median() or 20)
    rows, current, last_mid = [], [], None
    for _, w in words.iterrows():
        if last_mid is not None and w["mid"] - last_mid > line_h * 0.6:
            rows.append(current); current = []
        current.append(w); last_mid = w["mid"] if not current[:-1] else (last_mid + w["mid"]) / 2
    if current:
        rows.append(current)
    return "\n".join(" ".join(x["text"] for x in sorted(r, key=lambda x: x["left"])) for r in rows)


def ocr_page(path) -> tuple[str, float]:
    """OCR once with Tesseract; cache the word boxes so reruns are fast."""
    cache = OUTPUT / "ocr_cache" / f"{path.stem}.csv"
    if cache.exists():
        words = pd.read_csv(cache, keep_default_na=False)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["pdftoppm", "-r", "200", "-gray", "-png", "-singlefile", str(path), f"{tmp}/p"], check=True)
            data = pytesseract.image_to_data(Image.open(f"{tmp}/p.png"), output_type=pytesseract.Output.DATAFRAME)
        words = data[data["conf"] > 0].copy()
        words["text"] = [("" if pd.isna(t) else str(t)).strip() for t in words["text"]]
        words = words[words["text"] != ""][["left", "top", "width", "height", "conf", "text"]]
        cache.parent.mkdir(exist_ok=True)
        words.to_csv(cache, index=False)
    words["text"] = words["text"].astype(str)
    conf = float(words["conf"].mean()) if len(words) else 0.0
    return rows_from_words(words), conf


def read_text(path) -> tuple[str, str, float | None]:
    with pdfplumber.open(path) as pdf:
        text = pdf.pages[0].extract_text() or ""
    if len(text.strip()) > 40:
        return text, "pdf_text", None
    text, conf = ocr_page(path)
    return text, "ocr", conf


# ---------- value parsers ----------
def parse_amount(raw: str | None) -> tuple[float | None, str | None]:
    if not raw:
        return None, None
    s = raw.strip()
    cur = next((c for c in CODES if c in s), None)
    if cur is None:
        cur = next((code for sym, code in SYMBOLS.items() if sym in s), None)
    num = re.sub(r"[^\d.,]", "", s)
    if not num:
        return None, cur
    if re.search(r",\d{2}$", num):  # European 1.234,56
        num = num.replace(".", "").replace(",", ".")
    else:
        num = num.replace(",", "")
    try:
        return round(float(num), 2), cur
    except ValueError:
        return None, cur


def parse_date(raw: str | None, day_first: bool) -> str | None:
    if not raw:
        return None
    s = raw.strip().rstrip(".")
    fmts = ["%b %d, %Y", "%Y-%m-%d", "%d.%m.%Y", "%B %d, %Y"]
    fmts += ["%d/%m/%Y", "%m/%d/%Y"] if day_first else ["%m/%d/%Y", "%d/%m/%Y"]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date().isoformat()
        except ValueError:
            continue
    return None


def grab(patterns: list[str], text: str, flags=re.I) -> str | None:
    for p in patterns:
        p = p.replace(r"\s*", r"[ \t]*").replace(r"\s+", r"[ \t]+")  # label and value on the same line
        m = re.search(p, text, flags)
        if m:
            return m.group(1).strip()
    return None


AMT = r"((?:[A-Z]{3}\s*)?[$€£]?\s*[\d.,]+\d(?:\s*(?:[A-Z]{3}|€))?)"


# ---------- rules for layouts A-D ----------
def extract_fields(text: str, vendors: pd.DataFrame) -> dict:
    out: dict = {}
    tax_id = grab([r"(TX-\d{4}-\d{5})"], text)
    out["vendor_tax_id"] = tax_id
    header = " ".join(text.splitlines()[:3])
    match = process.extractOne(header, vendors["Vendor Name"].tolist(), scorer=fuzz.partial_ratio)
    out["vendor_name"] = match[0] if match else None

    two_col = re.search(r"Invoice Number\s+Purchase Order\s*\n\s*(\S+)\s+(\S+)", text)  # layout B
    out["invoice_number"] = two_col.group(1) if two_col else grab(
        [r"Invoice\s*#:?\s*([A-Z0-9-]{6,})", r"Invoice no\.?\s*:?\s*([A-Z0-9-]{6,})",
         r"(?:Invoice|Credit Memo) No\.\s*([A-Z0-9-]{6,})"], text)
    po = two_col.group(2) if two_col else grab(
        [r"PO Number:\s*(\S+)", r"Your PO:\s*(\S+)", r"Order ref\.\s*(\S+)"], text)
    out["po_number"] = None if (po is None or po.upper() in {"N/A", "-", "—", "NOT"}) else po

    country_us = "United States" in text
    b_dates = re.search(r"Issue Date\s+Payment Terms\s*\n\s*(\S+)\s+(\d+) days", text)
    b_due = re.search(r"Due Date\s+Cost Center\s*\n\s*(\S+)", text)
    inv_raw = b_dates.group(1) if b_dates else grab(
        [r"Invoice Date:\s*([A-Za-z]{3} \d{1,2}, \d{4})", r"^Date:\s*([\d/]+)", r"Invoice date\s*([\d.]+)"], text, re.I | re.M)
    due_raw = b_due.group(1) if b_due else grab(
        [r"Due Date:\s*([A-Za-z]{3} \d{1,2}, \d{4})", r"by\s*([\d/]{10})", r"Payable by\s*([\d.]+)"], text)
    out["invoice_date"] = parse_date(inv_raw, day_first=not country_us)
    out["due_date"] = parse_date(due_raw, day_first=not country_us)
    terms = b_dates.group(2) if b_dates else grab(
        [r"Net[ \t]+(\d{1,3})\b", r"within\s+(\d{1,3})\s+days", r"(\d{1,3})\s+days\s+net"], text)
    out["terms_days"] = int(terms) if terms else None

    total_raw = grab([r"TOTAL DUE\s*" + AMT, r"Amount Due\s*" + AMT, r"Total amount payable:\s*" + AMT,
                      r"Total \(gross\)\s*" + AMT], text)
    sub_raw = grab([r"Subtotal\s*" + AMT, r"Net amount\s*" + AMT, r"Net total\s*" + AMT], text)
    tax_raw = grab([r"(?:Sales tax|GST|VAT)[^\n]*?%\)?\s*" + AMT, r"^(?:GST|VAT|Sales tax)\s+" + AMT], text, re.I | re.M)
    out["total"], cur = parse_amount(total_raw)
    out["subtotal"], _ = parse_amount(sub_raw)
    out["tax"], _ = parse_amount(tax_raw) if tax_raw else (0.0 if out["subtotal"] is not None else None, None)
    out["currency"] = cur
    out["is_credit_memo"] = bool(re.search(r"credit memo", text, re.I))
    return out


def run_one(args):
    path, vendors = args
    text, source, conf = read_text(path)
    fields = extract_fields(text, vendors)
    return {"invoice_id": path.stem, "text_source": source, "ocr_confidence": conf} | fields


def main() -> None:
    vendors = pd.read_csv(OUTPUT / "vendor_master_addresses.csv")
    files = sorted(INVOICE_DIR.glob("*.pdf"))
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run_one, [(f, vendors) for f in files], chunksize=8))
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "extracted_rules.csv", index=False)
    print(df["text_source"].value_counts())
    print(df[FIELDS].notna().mean().round(3))


if __name__ == "__main__":
    main()
