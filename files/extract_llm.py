"""AI extractor: send each invoice PDF to Claude and get the fields back as JSON.

No templates and no OCR step: the model reads the PDF (text or scanned image)
directly, and a forced tool call guarantees the output matches the schema.
Results are cached per invoice in output/llm_cache/, so a rerun only calls the
API for invoices it has not seen yet.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
Run:
    python src/extract_llm.py                      # Claude Haiku 4.5, all 836 PDFs
    python src/extract_llm.py --limit 20           # quick test on 20 invoices
    python src/extract_llm.py --model claude-sonnet-4-6
Cost: about 3K input tokens per one-page invoice, so about $4 for all 836
invoices on Haiku 4.5 ($1 / $5 per million input / output tokens).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from common import INVOICE_DIR, OUTPUT

PROMPT = """You are an accounts payable clerk. Read this supplier invoice and record its fields
with the record_invoice tool. Rules:
- Copy invoice_number, po_number and vendor_tax_id exactly as printed (keep dashes/letters).
- po_number is the buyer's purchase order reference. Use null if none is given ("N/A", "-", "none").
- Dates as YYYY-MM-DD. For dates like 03/04/2025, use the supplier's country to decide the
  order (United States = month first, most other countries = day first).
- total is the final amount the buyer must pay (after tax, after any amounts already paid),
  not the subtotal. Amounts are plain numbers: 1.234,56 means 1234.56.
- currency is the ISO code (USD, EUR, GBP, CAD, SGD, INR). "$" alone means USD.
- terms_days is the number of days allowed to pay (e.g. "Net 30" -> 30).
- If a field is missing or unreadable, use null. Never guess."""

TOOL = {
    "name": "record_invoice",
    "description": "Record the header fields of one supplier invoice.",
    "input_schema": {
        "type": "object",
        "properties": {
            "vendor_name": {"type": ["string", "null"]},
            "vendor_tax_id": {"type": ["string", "null"]},
            "vendor_country": {"type": ["string", "null"]},
            "document_type": {"type": "string", "enum": ["invoice", "credit_memo"]},
            "invoice_number": {"type": ["string", "null"]},
            "invoice_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
            "due_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
            "po_number": {"type": ["string", "null"]},
            "currency": {"type": ["string", "null"]},
            "subtotal": {"type": ["number", "null"]},
            "tax": {"type": ["number", "null"]},
            "total": {"type": ["number", "null"]},
            "terms_days": {"type": ["integer", "null"]},
            "line_items": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "description": {"type": "string"}, "amount": {"type": ["number", "null"]}}},
            },
        },
        "required": ["vendor_name", "invoice_number", "invoice_date", "po_number", "currency", "total",
                     "document_type"],
    },
}


def extract_one(client, model: str, path) -> dict:
    import anthropic
    cache = OUTPUT / "llm_cache" / f"{path.stem}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    pdf_b64 = base64.standard_b64encode(path.read_bytes()).decode()
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=1024,
                tools=[TOOL],
                tool_choice={"type": "tool", "name": "record_invoice"},
                messages=[{"role": "user", "content": [
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64}},
                    {"type": "text", "text": PROMPT},
                ]}],
            )
            break
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as err:
            if attempt == 4:  # rate limits / overloads: back off and retry
                raise
            time.sleep(2 ** attempt * 3)
            print(f"retry {path.stem}: {err}")
    fields = next(block.input for block in msg.content if block.type == "tool_use")
    record = {"invoice_id": path.stem, "model": model, "input_tokens": msg.usage.input_tokens,
              "output_tokens": msg.usage.output_tokens, "fields": fields}
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(record, indent=1))
    return record


def to_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        f = r["fields"]
        rows.append({
            "invoice_id": r["invoice_id"], "text_source": "llm", "ocr_confidence": None,
            "vendor_tax_id": f.get("vendor_tax_id"), "vendor_name": f.get("vendor_name"),
            "invoice_number": f.get("invoice_number"), "invoice_date": f.get("invoice_date"),
            "due_date": f.get("due_date"), "po_number": f.get("po_number"), "currency": f.get("currency"),
            "subtotal": f.get("subtotal"), "tax": f.get("tax"), "total": f.get("total"),
            "terms_days": f.get("terms_days"), "is_credit_memo": f.get("document_type") == "credit_memo",
            "line_sum": sum((li.get("amount") or 0) for li in f.get("line_items") or []) or None,
            "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    import anthropic  # imported here so the rest of the project runs without it
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY first (console.anthropic.com > API keys).")
    client = anthropic.Anthropic()

    files = sorted(INVOICE_DIR.glob("*.pdf"))[: args.limit]
    records = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(extract_one, client, args.model, f): f for f in files}
        for k, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if k % 50 == 0:
                print(f"{k}/{len(files)} done")
    df = to_table(records).sort_values("invoice_id")
    df.to_csv(OUTPUT / "extracted_llm.csv", index=False)
    cost = df["input_tokens"].sum() / 1e6 * 1.0 + df["output_tokens"].sum() / 1e6 * 5.0
    print(f"{len(df)} invoices, {df['input_tokens'].sum():,} input tokens, ~${cost:.2f} at Haiku 4.5 prices")


if __name__ == "__main__":
    main()
