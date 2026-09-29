"""Shared helpers: paths, source tables, the test-period invoice set, vendor profiles."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "AP_Finance_Operations_Eisen.xlsx"
INVOICE_DIR = ROOT / "invoices"
OUTPUT = ROOT / "output"

BILL_TO = ["Crestline Holdings, Inc.", "Accounts Payable", "500 Harbor Blvd, Suite 1200", "Oakland, CA 94607"]

# Layout each vendor uses. A-D were available when the rule-based extractor was
# written. E is a "new supplier layout" held out to test how each method handles
# a format it has never seen.
KNOWN_LAYOUTS = ["A", "B", "C"]
EUROPEAN_LAYOUT = "D"
UNSEEN_LAYOUT = "E"

TAX = {  # (label, rate) by currency; goods only for USD
    "USD": ("Sales tax", 0.0825),
    "CAD": ("GST", 0.05),
    "EUR": ("VAT", 0.21),
    "GBP": ("VAT", 0.20),
    "SGD": ("GST", 0.09),
    "INR": ("GST", 0.18),
}
GOODS = {"Hardware", "Office Supplies", "Facilities"}

CITIES = {
    "United States": [("Austin", "TX 78701"), ("Denver", "CO 80202"), ("Columbus", "OH 43215"),
                      ("Raleigh", "NC 27601"), ("Phoenix", "AZ 85004"), ("Seattle", "WA 98101"),
                      ("Chicago", "IL 60601"), ("Atlanta", "GA 30303")],
    "Canada": [("Toronto", "ON M5H 2N2"), ("Vancouver", "BC V6B 1A1"), ("Calgary", "AB T2P 1J9")],
    "Germany": [("Berlin", "10115"), ("Munich", "80331"), ("Hamburg", "20095")],
    "Ireland": [("Dublin 2", "D02 X285"), ("Cork", "T12 W8DR"), ("Galway", "H91 E2K7")],
    "India": [("Bengaluru", "560001"), ("Pune", "411001"), ("Hyderabad", "500081")],
    "Singapore": [("Singapore", "018989"), ("Singapore", "048616")],
    "United Kingdom": [("London", "EC2A 4NE"), ("Manchester", "M1 1AE")],
}
STREETS = ["Market St", "Industrial Way", "Commerce Dr", "Park Ave", "Innovation Blvd",
           "Harbour Rd", "King St", "Station Rd", "Lake View Dr", "Enterprise Pkwy"]


def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


def load_tables() -> dict[str, pd.DataFrame]:
    sheets = ["Invoices", "Vendors", "Purchase Orders", "FX Rates", "Payments", "Review Log"]
    tables = pd.read_excel(DATA, sheet_name=sheets)
    for col in ["Invoice Date", "Received Date", "Due Date", "Approval Date"]:
        tables["Invoices"][col] = pd.to_datetime(tables["Invoices"][col])
    return tables


def vendor_profiles(vendors: pd.DataFrame) -> pd.DataFrame:
    """Add a remit-to address, tax ID and invoice layout to each vendor.

    The vendor master has no addresses, and a few vendor names appear twice under
    different IDs, so the invoice needs a second identifier (tax ID) to match.
    """
    rows = []
    for _, v in vendors.iterrows():
        h = stable_int(v["Vendor ID"])
        city, post = CITIES[v["Country"]][h % len(CITIES[v["Country"]])]
        street = f"{100 + h % 8900} {STREETS[(h // 7) % len(STREETS)]}"
        tax_id = f"TX-{v['Vendor ID'][2:]}-{h % 90000 + 10000}"
        if v["Default Currency"] == "EUR":
            layout = EUROPEAN_LAYOUT
        elif (h // 13) % 5 == 0:
            layout = UNSEEN_LAYOUT
        else:
            layout = KNOWN_LAYOUTS[(h // 11) % 3]
        rows.append({"Vendor ID": v["Vendor ID"], "Street": street, "City": city, "Postcode": post,
                     "Tax ID": tax_id, "Layout": layout})
    return vendors.merge(pd.DataFrame(rows), on="Vendor ID")


def test_period(invoices: pd.DataFrame, fraction: float = 0.20) -> pd.DataFrame:
    """Same chronological 80/20 split as the AP model: last 20% by received date."""
    ordered = invoices.sort_values(["Received Date", "Invoice ID"]).reset_index(drop=True)
    cut = int(len(ordered) * (1 - fraction))
    return ordered.iloc[cut:].reset_index(drop=True)
