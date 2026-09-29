"""Turn the test-period invoice records into realistic PDF invoices.

Each vendor always uses the same layout (like real suppliers). Supplier-portal
invoices are digital PDFs; about 40% of emailed invoices are scans (image-only
PDFs with skew and noise). EDI invoices arrive as structured data, so they get
no PDF. Ground truth for every field is saved to output/ground_truth.csv.

Run:  python src/generate_invoices.py
"""
from __future__ import annotations

import io
import json
import random
import subprocess
import tempfile
from pathlib import Path

import img2pdf
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from common import BILL_TO, GOODS, INVOICE_DIR, OUTPUT, TAX, load_tables, stable_int, test_period, vendor_profiles

W, H = LETTER
SCAN_SHARE_EMAIL = 0.40

CATALOG = {
    "Hardware": ["Laptop docking station", "27in monitor", "Network switch 24-port", "Rack server memory kit", "Wireless keyboard set"],
    "Marketing": ["Campaign creative services", "Paid social management", "Event booth design", "Content production"],
    "Recruiting": ["Contract recruiter hours", "Job board posting package", "Candidate background checks"],
    "Facilities": ["Janitorial services", "HVAC maintenance", "Office furniture", "Security badge readers"],
    "Travel": ["Airfare - team offsite", "Hotel accommodation", "Ground transportation"],
    "Telecommunications": ["Mobile plan - corporate lines", "Fiber internet circuit", "Conference bridge service"],
    "Office Supplies": ["Printer paper (cases)", "Toner cartridges", "Desk supplies bundle"],
    "Insurance": ["General liability premium", "Cyber insurance premium", "Property coverage installment"],
    "Cloud Infrastructure": ["Compute instances", "Object storage", "Managed database service"],
    "Software": ["Annual license subscription", "User seats", "Premium support plan"],
    "Legal Services": ["Contract review", "Regulatory advisory hours", "Filing fees"],
    "Consulting": ["Advisory hours", "Process assessment", "Workshop facilitation"],
    "Logistics": ["Freight - LTL", "Warehouse handling", "Last-mile delivery"],
}
CUR_SYMBOL = {"USD": "$", "CAD": "C$", "GBP": "£", "SGD": "S$", "INR": "Rs.", "EUR": "€"}


# ---------- number and date formats ----------
def fmt_us(x: float) -> str:
    return f"{x:,.2f}"


def fmt_eu(x: float) -> str:
    return f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_in(x: float) -> str:
    whole, frac = f"{x:.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{whole}.{frac}"


def money(x: float, cur: str, layout: str) -> str:
    num = fmt_in(x) if cur == "INR" else fmt_us(x)
    if layout == "A":
        return f"${num}" if cur == "USD" else f"{cur} {num}"
    if layout == "B":
        return f"{num} {cur}"
    if layout == "C":
        return f"{cur} {num}"
    if layout == "D":
        return f"{fmt_eu(x)} €" if cur == "EUR" else f"{fmt_eu(x)} {cur}"
    return f"{CUR_SYMBOL[cur]}{num}"  # E


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def date_str(d: pd.Timestamp, layout: str, country: str) -> str:
    if layout == "A":
        return d.strftime("%b %d, %Y")
    if layout == "B":
        return d.strftime("%Y-%m-%d")
    if layout == "C":  # US vendors write month first, others day first
        return d.strftime("%m/%d/%Y") if country == "United States" else d.strftime("%d/%m/%Y")
    if layout == "D":
        return d.strftime("%d.%m.%Y")
    return f"{ordinal(d.day)} {d.strftime('%B %Y')}"


def po_str(po: str, layout: str) -> str:
    return po.replace("-", "") if layout == "E" else po


def vendor_display(name: str, layout: str) -> str:
    if layout == "B":
        return name.upper()
    if layout == "C":
        for suffix in [" LLC", " Inc.", " Ltd."]:
            if name.endswith(suffix):
                return name[: -len(suffix)] + "," + suffix
    if layout == "E":
        return name.replace(" Group", " Grp").replace(" Inc.", " Incorporated")
    return name


# ---------- line items ----------
def build_lines(total: float, category: str, cur: str, rng: random.Random):
    taxable = category in GOODS or cur != "USD"
    label, rate = TAX[cur]
    rate = rate if taxable else 0.0
    subtotal = round(total / (1 + rate), 2)
    tax = round(total - subtotal, 2)
    items = CATALOG.get(category, ["Professional services"])
    n = min(len(items), rng.choice([1, 1, 2, 2, 3, 4]))
    names = rng.sample(items, n)
    weights = np.array([rng.uniform(0.5, 1.5) for _ in range(n)])
    amounts = np.round(subtotal * weights / weights.sum(), 2)
    amounts[-1] = round(subtotal - amounts[:-1].sum(), 2)
    lines = []
    for name, amt in zip(names, amounts):
        qty = rng.choice([1, 1, 2, 4, 5, 10, 12]) if amt > 50 else 1
        unit = round(amt / qty, 2)
        if round(unit * qty, 2) != round(amt, 2):
            qty, unit = 1, float(amt)
        lines.append({"description": name, "qty": qty, "unit": unit, "amount": float(amt)})
    return lines, subtotal, tax, label, rate


# ---------- drawing ----------
def draw_invoice(path: Path, rec: dict) -> None:
    L = rec["layout"]
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setTitle(f"{rec['doc_title']} {rec['invoice_number']}")
    cur, country = rec["currency"], rec["country"]
    d_inv = date_str(rec["invoice_date"], L, country)
    d_due = date_str(rec["due_date"], L, country)
    m = lambda x: money(x, cur, L)  # noqa: E731
    vname = vendor_display(rec["vendor_name"], L)
    addr = [rec["street"], f"{rec['city']} {rec['postcode']}", rec["country"]]
    po = po_str(rec["po_number"], L) if rec["po_number"] else None

    if L == "A":  # classic: vendor top-left, meta box top-right
        c.setFont("Helvetica-Bold", 16); c.drawString(50, H - 60, vname)
        c.setFont("Helvetica", 9)
        for k, line in enumerate(addr + [f"Tax ID: {rec['tax_id']}"]):
            c.drawString(50, H - 76 - 12 * k, line)
        c.setFont("Helvetica-Bold", 22); c.drawRightString(W - 50, H - 60, rec["doc_title"].upper())
        c.setFont("Helvetica", 10)
        meta = [("Invoice #", rec["invoice_number"]), ("Invoice Date", d_inv),
                ("PO Number", po or "N/A"), ("Terms", f"Net {rec['terms_days']}"), ("Due Date", d_due)]
        for k, (a, b) in enumerate(meta):
            c.drawRightString(W - 160, H - 90 - 14 * k, a + ":"); c.drawRightString(W - 50, H - 90 - 14 * k, b)
        y = H - 175
        c.setFont("Helvetica-Bold", 10); c.drawString(50, y, "Bill To"); c.setFont("Helvetica", 10)
        for k, line in enumerate(BILL_TO + [f"Attn: {rec['department']} ({rec['cost_center']})"]):
            c.drawString(50, y - 13 * (k + 1), line)
        y -= 110
        cols = [(50, "Description"), (330, "Qty"), (400, "Unit Price"), (W - 50, "Amount")]
        c.setFillColor(colors.HexColor("#1f3b5a")); c.rect(45, y - 4, W - 90, 16, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 9)
        for x, t in cols:
            (c.drawRightString if x == W - 50 else c.drawString)(x, y, t)
        c.setFillColor(colors.black); c.setFont("Helvetica", 9)
        for ln in rec["lines"]:
            y -= 16
            c.drawString(50, y, ln["description"]); c.drawString(330, y, str(ln["qty"]))
            c.drawString(400, y, m(ln["unit"])); c.drawRightString(W - 50, y, m(ln["amount"]))
        y -= 30
        tot = [("Subtotal", m(rec["subtotal"]))]
        if rec["tax"]:
            tot.append((f"{rec['tax_label']} ({rec['tax_rate']:.2%})", m(rec["tax"])))
        for a, b in tot:
            c.drawRightString(W - 160, y, a); c.drawRightString(W - 50, y, b); y -= 14
        c.setFont("Helvetica-Bold", 11); c.drawRightString(W - 160, y - 4, "TOTAL DUE"); c.drawRightString(W - 50, y - 4, m(rec["total"]))
        c.setFont("Helvetica", 8); c.drawString(50, 60, rec["note"])

    elif L == "B":  # modern: colored band, two-column labels, big amount due
        c.setFillColor(colors.HexColor("#0e7c7b")); c.rect(0, H - 90, W, 90, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 18); c.drawString(40, H - 50, vname)
        c.setFont("Helvetica", 9); c.drawString(40, H - 66, " | ".join(addr)); c.drawString(40, H - 78, f"VAT/Tax Reg. {rec['tax_id']}")
        c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 14); c.drawString(40, H - 125, rec["doc_title"])
        c.setFont("Helvetica", 9)
        left = [("Invoice Number", rec["invoice_number"]), ("Issue Date", d_inv), ("Due Date", d_due)]
        right = [("Purchase Order", po or "-"), ("Payment Terms", f"{rec['terms_days']} days"), ("Cost Center", rec["cost_center"])]
        for k, ((a, b), (e, f)) in enumerate(zip(left, right)):
            c.setFillColor(colors.grey); c.drawString(40, H - 150 - 26 * k, a); c.drawString(320, H - 150 - 26 * k, e)
            c.setFillColor(colors.black); c.drawString(40, H - 161 - 26 * k, b); c.drawString(320, H - 161 - 26 * k, f)
        y = H - 250
        c.setFont("Helvetica-Bold", 9); c.drawString(40, y, "Billed to"); c.setFont("Helvetica", 9)
        c.drawString(40, y - 12, ", ".join(BILL_TO))
        y -= 45
        c.setFont("Helvetica-Bold", 9)
        c.drawString(40, y, "Item"); c.drawRightString(380, y, "Quantity"); c.drawRightString(470, y, "Rate"); c.drawRightString(W - 40, y, "Line total")
        c.line(40, y - 4, W - 40, y - 4); c.setFont("Helvetica", 9)
        for ln in rec["lines"]:
            y -= 16
            c.drawString(40, y, ln["description"]); c.drawRightString(380, y, str(ln["qty"]))
            c.drawRightString(470, y, fmt_us(ln["unit"])); c.drawRightString(W - 40, y, m(ln["amount"]))
        y -= 26
        c.drawRightString(470, y, "Net amount"); c.drawRightString(W - 40, y, m(rec["subtotal"]))
        if rec["tax"]:
            y -= 14; c.drawRightString(470, y, rec["tax_label"]); c.drawRightString(W - 40, y, m(rec["tax"]))
        y -= 34
        c.setFillColor(colors.HexColor("#e6f4f3")); c.rect(300, y - 10, W - 340, 34, fill=1, stroke=0)
        c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 12)
        c.drawString(310, y + 3, "Amount Due"); c.drawRightString(W - 50, y + 3, m(rec["total"]))
        c.setFont("Helvetica", 8); c.drawString(40, 50, rec["note"])

    elif L == "C":  # plain letter style, labels inline in sentences
        c.setFont("Times-Bold", 14); c.drawString(72, H - 72, vname)
        c.setFont("Times-Roman", 10)
        c.drawString(72, H - 88, ", ".join(addr)); c.drawString(72, H - 100, f"Tax registration: {rec['tax_id']}")
        y = H - 140
        body = [f"{rec['doc_title']} No. {rec['invoice_number']}", f"Date: {d_inv}", "",
                "To: " + BILL_TO[0] + ", " + BILL_TO[1], "     " + BILL_TO[2] + ", " + BILL_TO[3], ""]
        body.append(f"Your PO: {po}" if po else "Your PO: not provided")
        body += [f"Department: {rec['department']}", ""]
        for line in body:
            c.drawString(72, y, line); y -= 14
        for ln in rec["lines"]:
            c.drawString(90, y, f"- {ln['description']}, {ln['qty']} x {fmt_us(ln['unit'])}")
            c.drawRightString(W - 72, y, fmt_us(ln["amount"])); y -= 14
        y -= 8
        c.drawString(90, y, "Subtotal"); c.drawRightString(W - 72, y, fmt_us(rec["subtotal"])); y -= 14
        if rec["tax"]:
            c.drawString(90, y, f"{rec['tax_label']} at {rec['tax_rate'] * 100:g}%"); c.drawRightString(W - 72, y, fmt_us(rec["tax"])); y -= 14
        y -= 10
        c.setFont("Times-Bold", 11)
        c.drawString(72, y, f"Total amount payable: {m(rec['total'])}"); y -= 18
        c.setFont("Times-Roman", 10)
        c.drawString(72, y, f"Payment due within {rec['terms_days']} days, by {d_due}."); y -= 28
        c.drawString(72, y, rec["note"])

    elif L == "D":  # European: dotted dates, 1.234,56 € amounts
        c.setFont("Helvetica-Bold", 13); c.drawString(50, H - 55, vname)
        c.setFont("Helvetica", 8); c.drawString(50, H - 68, " · ".join(addr) + f" · VAT {rec['tax_id']}")
        c.setFont("Helvetica", 9)
        for k, line in enumerate(BILL_TO):
            c.drawString(50, H - 110 - 11 * k, line)
        c.setFont("Helvetica-Bold", 16); c.drawString(50, H - 185, rec["doc_title"])
        c.setFont("Helvetica", 9)
        meta = [("Invoice no.", rec["invoice_number"]), ("Invoice date", d_inv), ("Order ref.", po or "—"),
                ("Terms", f"{rec['terms_days']} days net"), ("Payable by", d_due)]
        for k, (a, b) in enumerate(meta):
            c.drawString(380, H - 110 - 13 * k, a); c.drawRightString(W - 50, H - 110 - 13 * k, b)
        y = H - 225
        c.setFont("Helvetica-Bold", 9)
        c.drawString(50, y, "Pos."); c.drawString(80, y, "Description"); c.drawRightString(400, y, "Qty"); c.drawRightString(W - 50, y, "Net")
        c.setFont("Helvetica", 9)
        for k, ln in enumerate(rec["lines"], 1):
            y -= 15
            c.drawString(50, y, str(k)); c.drawString(80, y, ln["description"])
            c.drawRightString(400, y, str(ln["qty"])); c.drawRightString(W - 50, y, money(ln["amount"], cur, "D"))
        y -= 12; c.line(300, y, W - 50, y); y -= 14
        c.drawString(300, y, "Net total"); c.drawRightString(W - 50, y, m(rec["subtotal"])); y -= 13
        c.drawString(300, y, f"VAT {rec['tax_rate'] * 100:g}%"); c.drawRightString(W - 50, y, m(rec["tax"])); y -= 15
        c.setFont("Helvetica-Bold", 10); c.drawString(300, y, "Total (gross)"); c.drawRightString(W - 50, y, m(rec["total"]))
        c.setFont("Helvetica", 8); c.drawString(50, 60, rec["note"])

    else:  # E: the unseen layout, labels the rule-based parser has never seen
        c.setFont("Courier-Bold", 13); c.drawCentredString(W / 2, H - 50, vname)
        c.setFont("Courier", 9); c.drawCentredString(W / 2, H - 64, ", ".join(addr))
        c.drawCentredString(W / 2, H - 76, f"Company reg / tax: {rec['tax_id']}")
        c.line(50, H - 86, W - 50, H - 86)
        c.setFont("Courier-Bold", 11); c.drawString(50, H - 110, "STATEMENT OF CHARGES" if rec["doc_title"] == "Invoice" else rec["doc_title"].upper())
        c.setFont("Courier", 9)
        meta = [("Bill No.", rec["invoice_number"]), ("Issued", d_inv), ("Customer Ref / PO", po or "(none)"),
                ("Settlement", f"Due in {rec['terms_days']} days"), ("Settle by", d_due)]
        for k, (a, b) in enumerate(meta):
            c.drawString(50, H - 130 - 12 * k, f"{a:<20}{b}")
        c.drawString(330, H - 130, "Customer:")
        for k, line in enumerate(BILL_TO):
            c.drawString(330, H - 142 - 12 * k, line)
        y = H - 215
        c.drawString(50, y, "-" * 86); y -= 12
        for ln in rec["lines"]:
            c.drawString(50, y, f"{ln['description'][:34]:<36}{ln['qty']:>4} @ {fmt_us(ln['unit']):>12}")
            c.drawRightString(W - 50, y, money(ln["amount"], cur, "E")); y -= 12
        c.drawString(50, y, "-" * 86); y -= 14
        c.drawString(300, y, "Charges"); c.drawRightString(W - 50, y, money(rec["subtotal"], cur, "E")); y -= 12
        if rec["tax"]:
            c.drawString(300, y, f"{rec['tax_label']}"); c.drawRightString(W - 50, y, money(rec["tax"], cur, "E")); y -= 12
        c.drawString(300, y, "Paid to date"); c.drawRightString(W - 50, y, money(0.0, cur, "E")); y -= 16
        c.setFont("Courier-Bold", 10); c.drawString(300, y, "Balance Due"); c.drawRightString(W - 50, y, money(rec["total"], cur, "E"))
        c.setFont("Courier", 8); c.drawString(50, 60, rec["note"])
    c.showPage(); c.save()


def make_scan(pdf_path: Path, seed: int) -> None:
    """Rasterize a digital PDF and degrade it like a phone/office scan."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "150", "-gray", "-png", "-singlefile", str(pdf_path), f"{tmp}/p"], check=True)
        img = Image.open(f"{tmp}/p.png").convert("L")
    img = img.rotate(float(rng.uniform(-1.5, 1.5)), expand=True, fillcolor=255, resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr * rng.uniform(0.85, 0.95) + rng.uniform(10, 25) + rng.normal(0, 9, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 0.7))))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=int(rng.integers(45, 70)))
    pdf_path.write_bytes(img2pdf.convert(buf.getvalue(), layout_fun=img2pdf.get_fixed_dpi_layout_fun((150, 150))))


def main() -> None:
    tables = load_tables()
    vendors = vendor_profiles(tables["Vendors"]).set_index("Vendor ID")
    test = test_period(tables["Invoices"])
    INVOICE_DIR.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)
    for old in INVOICE_DIR.glob("*.pdf"):
        old.unlink()

    truth, edi = [], []
    for _, inv in test.iterrows():
        v = vendors.loc[inv["Vendor ID"]]
        rng = random.Random(stable_int(inv["Invoice ID"]))
        category = inv["Description"].replace(" invoice", "")
        lines, subtotal, tax, tax_label, rate = build_lines(float(inv["Invoice Amount"]), category, inv["Currency"], rng)
        po = inv["PO ID"] if isinstance(inv["PO ID"], str) else ""
        doc_title = {"Credit memo": "Credit Memo"}.get(inv["Invoice Type"], "Invoice")
        note = {"Recurring": f"Service period: {inv['Invoice Date'].strftime('%B %Y')}. Recurring charge.",
                "Prepayment": "Advance payment - services to be delivered next period.",
                "Credit memo": f"Credit issued against prior billing. Reference {inv['Invoice Number']}."}.get(
            inv["Invoice Type"], "Thank you for your business. Please include the invoice number with payment.")
        rec = {
            "invoice_id": inv["Invoice ID"], "channel": inv["Submitted Channel"], "layout": v["Layout"],
            "vendor_id": inv["Vendor ID"], "vendor_name": v["Vendor Name"], "tax_id": v["Tax ID"],
            "street": v["Street"], "city": v["City"], "postcode": v["Postcode"], "country": v["Country"],
            "invoice_number": inv["Invoice Number"], "invoice_date": inv["Invoice Date"].normalize(),
            "due_date": inv["Due Date"].normalize(), "terms_days": int(inv["Payment Terms Days"]),
            "po_number": po, "currency": inv["Currency"], "total": round(float(inv["Invoice Amount"]), 2),
            "subtotal": subtotal, "tax": tax, "tax_label": tax_label, "tax_rate": rate, "lines": lines,
            "department": inv["Department"], "cost_center": inv["Cost Center"], "doc_title": doc_title, "note": note,
        }
        if inv["Submitted Channel"] == "EDI":
            rec["file"], rec["scanned"] = "", False
            edi.append({k: rec[k] for k in ["invoice_id", "vendor_id", "invoice_number", "po_number", "currency",
                                             "total", "subtotal", "tax", "terms_days"]}
                       | {"invoice_date": rec["invoice_date"].date().isoformat(), "due_date": rec["due_date"].date().isoformat(),
                          "vendor_tax_id": rec["tax_id"], "vendor_name": rec["vendor_name"]})
        else:
            path = INVOICE_DIR / f"{inv['Invoice ID']}.pdf"
            draw_invoice(path, rec)
            scanned = inv["Submitted Channel"] == "Email" and rng.random() < SCAN_SHARE_EMAIL
            if scanned:
                make_scan(path, stable_int(inv["Invoice ID"]) % 2**32)
            rec["file"], rec["scanned"] = path.name, scanned
        truth.append({k: val for k, val in rec.items() if k not in {"lines", "street", "city", "postcode", "note"}}
                     | {"line_count": len(lines)})

    gt = pd.DataFrame(truth)
    gt.to_csv(OUTPUT / "ground_truth.csv", index=False)
    (OUTPUT / "edi_feed.json").write_text(json.dumps(edi, indent=1))
    vendors.reset_index()[["Vendor ID", "Vendor Name", "Street", "City", "Postcode", "Country", "Tax ID", "Layout"]].to_csv(
        OUTPUT / "vendor_master_addresses.csv", index=False)
    print(gt.groupby(["channel", "scanned"]).size())
    print(gt.groupby("layout").size())


if __name__ == "__main__":
    main()
