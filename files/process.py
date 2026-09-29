"""Match extracted invoices to vendors and POs, run the AP controls, decide
auto-approve vs review, simulate the review queue, and score everything
against ground truth.

Compares up to three ways of reading the invoices:
  perfect  - true field values (upper bound: how good the controls themselves are)
  rules    - pdfplumber / Tesseract OCR + hand-written rules (output/extracted_rules.csv)
  llm      - Claude reading the PDF directly (output/extracted_llm.csv, if it exists)

Run:  python src/process.py
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process as rf_process

from common import OUTPUT, load_tables

sys.path.insert(0, str(OUTPUT.parent / "src"))
import ap_modeling_pipeline as ap  # noqa: E402  (Eisen's AP risk model + queue simulation)

# ---- assumptions (edit here) ----
KEY_MINUTES = 6.0          # minutes for a clerk to key + match one invoice by hand
DATA_CHECK_HOURS = 0.10    # fixing fields the extractor could not read (6 min)
OCR_MIN_CONFIDENCE = 80.0  # below this, a scanned page goes to a data check
REVIEWERS = 2
CONTROL_COLUMNS = ["Missing PO", "Invalid PO", "PO Vendor Mismatch", "Currency Mismatch", "Invoice Over PO",
                   "Payment Terms Mismatch", "Inactive Vendor", "Duplicate Invoice", "High-Risk Vendor"]
PO_BLANKS = {None, "NA", "NONE", "NOTPROVIDED", "NULL"}  # "N/A", "-", "(none)" etc. after canon()
FIELDS = ["vendor", "invoice_number", "invoice_date", "due_date", "po_number", "currency", "total", "terms_days"]


def canon(value) -> str | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


# ---------------- loading ----------------
def load_inputs():
    tables = load_tables()
    truth = pd.read_csv(OUTPUT / "ground_truth.csv", dtype={"po_number": str})
    truth["file"] = truth["file"].fillna("")
    vendors = pd.read_csv(OUTPUT / "vendor_master_addresses.csv").merge(
        tables["Vendors"][["Vendor ID", "Default Currency", "Default Terms Days", "Risk Tier", "Active Flag"]],
        on="Vendor ID")
    edi = pd.DataFrame(json.loads((OUTPUT / "edi_feed.json").read_text()))
    return tables, truth, vendors, edi


def perfect_extraction(truth: pd.DataFrame) -> pd.DataFrame:
    pdf = truth[truth["file"] != ""]
    return pd.DataFrame({
        "invoice_id": pdf["invoice_id"], "text_source": "perfect", "ocr_confidence": np.nan,
        "vendor_tax_id": pdf["tax_id"], "vendor_name": pdf["vendor_name"], "invoice_number": pdf["invoice_number"],
        "invoice_date": pd.to_datetime(pdf["invoice_date"]).dt.date.astype(str),
        "due_date": pd.to_datetime(pdf["due_date"]).dt.date.astype(str), "po_number": pdf["po_number"],
        "currency": pdf["currency"], "subtotal": pdf["subtotal"], "tax": pdf["tax"], "total": pdf["total"],
        "terms_days": pdf["terms_days"], "is_credit_memo": pdf["doc_title"].eq("Credit Memo"),
    })


def edi_records(edi: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "invoice_id": edi["invoice_id"], "text_source": "edi", "ocr_confidence": np.nan,
        "vendor_tax_id": edi["vendor_tax_id"], "vendor_name": edi["vendor_name"],
        "invoice_number": edi["invoice_number"], "invoice_date": edi["invoice_date"], "due_date": edi["due_date"],
        "po_number": edi["po_number"].replace("", None), "currency": edi["currency"], "subtotal": edi["subtotal"],
        "tax": edi["tax"], "total": edi["total"], "terms_days": edi["terms_days"], "is_credit_memo": False,
    })


# ---------------- matching ----------------
def resolve_vendor(row, vendors: pd.DataFrame, by_tax: dict, name_counts: pd.Series) -> tuple[str | None, str]:
    tax = canon(row.get("vendor_tax_id"))
    if tax:
        hit = by_tax.get(tax) or next((v for k, v in by_tax.items() if k in tax), None)  # e.g. "VAT TX-..."
        if hit:
            return hit, "tax_id"
    name = row.get("vendor_name")
    if isinstance(name, str) and name.strip():
        best = rf_process.extractOne(name, vendors["Vendor Name"].tolist(), scorer=fuzz.token_sort_ratio)
        if best and best[1] >= 90 and name_counts[best[0]] == 1:  # ambiguous names need a tax ID
            return vendors.iloc[best[2]]["Vendor ID"], "name"
    return None, "unmatched"


def run_controls(ext: pd.DataFrame, tables, vendors: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    by_tax = {canon(t): v for t, v in zip(vendors["Tax ID"], vendors["Vendor ID"])}
    name_counts = vendors["Vendor Name"].value_counts()
    vinfo = vendors.set_index("Vendor ID")
    pos = tables["Purchase Orders"].copy()
    pos["key"] = pos["PO ID"].map(canon)
    po_by_key = pos.set_index("key")

    df = ext.merge(truth[["invoice_id", "vendor_id", "layout", "scanned", "channel"]], on="invoice_id", how="left")
    inv = tables["Invoices"].set_index("Invoice ID")
    df["received"] = df["invoice_id"].map(inv["Received Date"])
    df = df.sort_values(["received", "invoice_id"]).reset_index(drop=True)

    res = df.apply(lambda r: resolve_vendor(r, vendors, by_tax, name_counts), axis=1)
    df["matched_vendor_id"] = [r[0] for r in res]
    df["vendor_match_method"] = [r[1] for r in res]
    df["po_key"] = [None if canon(x) in PO_BLANKS else canon(x) for x in df["po_number"]]
    df["po_key"] = df["po_key"].astype(object).where(df["po_key"].notna(), None)
    po_row = df["po_key"].map(lambda k: po_by_key.loc[k] if k in po_by_key.index else None)
    df["PO Vendor ID"] = [p["Vendor ID"] if p is not None else None for p in po_row]
    df["PO Amount"] = [p["PO Amount"] if p is not None else np.nan for p in po_row]
    df["PO Currency"] = [p["Currency"] if p is not None else None for p in po_row]
    v = df["matched_vendor_id"].map(lambda x: vinfo.loc[x] if x in vinfo.index else None)

    df["Missing PO"] = df["po_key"].isna().astype(int)
    df["Invalid PO"] = (df["po_key"].notna() & df["PO Amount"].isna()).astype(int)
    df["PO Vendor Mismatch"] = (df["PO Vendor ID"].notna() & df["matched_vendor_id"].notna()
                                & df["PO Vendor ID"].ne(df["matched_vendor_id"])).astype(int)
    df["Currency Mismatch"] = (df["PO Currency"].notna() & df["currency"].notna()
                               & df["currency"].ne(df["PO Currency"])).astype(int)
    df["Invoice Over PO"] = (df["PO Amount"].notna() & df["currency"].eq(df["PO Currency"])
                             & (pd.to_numeric(df["total"], errors="coerce") > df["PO Amount"] + 0.005)).astype(int)
    default_terms = [x["Default Terms Days"] if x is not None else np.nan for x in v]
    df["Payment Terms Mismatch"] = (pd.notna(default_terms) & pd.to_numeric(df["terms_days"], errors="coerce").notna()
                                    & pd.to_numeric(df["terms_days"], errors="coerce").ne(default_terms)).astype(int)
    df["Inactive Vendor"] = [int(x is not None and str(x["Active Flag"]).strip().lower() != "yes") for x in v]
    df["High-Risk Vendor"] = [int(x is not None and x["Risk Tier"] == "High") for x in v]

    # Duplicate: same vendor + invoice number already posted (ERP history) or already processed in this batch.
    test_ids = set(truth["invoice_id"])
    hist = tables["Invoices"][~tables["Invoices"]["Invoice ID"].isin(test_ids)]
    seen = {(vid, canon(n)) for vid, n in zip(hist["Vendor ID"], hist["Invoice Number"])}
    dup = []
    for vid, num in zip(df["matched_vendor_id"], df["invoice_number"]):
        key = (vid, canon(num))
        dup.append(int(vid is not None and key[1] is not None and key in seen))
        seen.add(key)
    df["Duplicate Invoice"] = dup

    # Guardrails: is the extracted data complete and internally consistent?
    total = pd.to_numeric(df["total"], errors="coerce")
    sub = pd.to_numeric(df["subtotal"], errors="coerce")
    tax = pd.to_numeric(df["tax"], errors="coerce").fillna(0)
    inv_d = pd.to_datetime(df["invoice_date"], errors="coerce")
    due_d = pd.to_datetime(df["due_date"], errors="coerce")
    terms = pd.to_numeric(df["terms_days"], errors="coerce")
    checks = pd.DataFrame({
        "Missing field": df[["invoice_number", "invoice_date", "total", "currency", "terms_days"]].isna().any(axis=1),
        "Vendor not matched": df["matched_vendor_id"].isna(),
        "Totals don't add up": sub.notna() & total.notna() & ((sub + tax - total).abs() > 0.02),
        "Due date != date + terms": inv_d.notna() & due_d.notna() & terms.notna()
                                    & (((due_d - inv_d).dt.days - terms).abs() > 1),
        "Low OCR confidence": pd.to_numeric(df["ocr_confidence"], errors="coerce") < OCR_MIN_CONFIDENCE,
    })
    df["Data Check Reasons"] = checks.apply(lambda r: "; ".join(c for c in checks.columns if r[c]), axis=1)
    df["Needs Data Check"] = checks.any(axis=1).astype(int)
    df["Control Flags"] = df[CONTROL_COLUMNS].apply(lambda r: "; ".join(c for c in CONTROL_COLUMNS if r[c]), axis=1)
    df["Control Flag Count"] = df[CONTROL_COLUMNS].sum(axis=1)
    # Bad or missing data goes to a person first (a blank PO field could be a reading
    # error, not a real missing PO). Clean data then runs through the controls.
    df["Decision"] = np.select(
        [df["Needs Data Check"] == 1, df["Control Flag Count"] > 0],
        ["Data check", "Exception review"], "Auto-approve")
    return df


# ---------------- scoring ----------------
def field_accuracy(df: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    t = truth.set_index("invoice_id")
    d = df[df["text_source"].ne("edi")].set_index("invoice_id")
    t = t.loc[d.index]
    ok = pd.DataFrame(index=d.index)
    ok["vendor"] = d["matched_vendor_id"].eq(t["vendor_id"])
    ok["invoice_number"] = [canon(a) == canon(b) and canon(a) is not None for a, b in zip(d["invoice_number"], t["invoice_number"])]
    ok["invoice_date"] = pd.to_datetime(d["invoice_date"], errors="coerce").eq(pd.to_datetime(t["invoice_date"]).dt.normalize())
    ok["due_date"] = pd.to_datetime(d["due_date"], errors="coerce").eq(pd.to_datetime(t["due_date"]).dt.normalize())
    ok["po_number"] = [(None if canon(a) in PO_BLANKS else canon(a)) == canon(b) for a, b in zip(d["po_number"], t["po_number"])]
    ok["currency"] = d["currency"].eq(t["currency"])
    ok["total"] = (pd.to_numeric(d["total"], errors="coerce") - t["total"]).abs() < 0.01
    ok["terms_days"] = pd.to_numeric(d["terms_days"], errors="coerce").eq(t["terms_days"])
    ok["all_fields"] = ok[FIELDS].all(axis=1)
    ok["layout_group"] = np.where(t["layout"].eq("E"), "New layout (E)", "Known layouts (A-D)")
    ok["format"] = np.where(t["scanned"], "Scanned", "Digital PDF")
    return ok


def decision_metrics(df: pd.DataFrame, tables) -> dict:
    inv = tables["Invoices"].set_index("Invoice ID")
    true_exc = df["invoice_id"].map(inv["Manual Review Required"]).eq("Yes")
    auto = df["Decision"].eq("Auto-approve")
    n = len(df)
    # Approval time: auto-approved invoices clear the day they arrive; anything a person
    # touches keeps the approval time observed in the source data.
    lag = df["invoice_id"].map((inv["Approval Date"] - inv["Received Date"]).dt.total_seconds() / 86400)
    new_lag = np.where(auto, 0.0, lag)
    return {
        "Avg days to approve": float(np.nanmean(new_lag)),
        "Invoices": n,
        "Auto-approved (touchless)": int(auto.sum()),
        "Touchless rate": auto.mean(),
        "Exception review": int(df["Decision"].eq("Exception review").sum()),
        "Data check": int(df["Decision"].eq("Data check").sum()),
        "True exceptions": int(true_exc.sum()),
        "Exceptions caught": int((true_exc & ~auto).sum()),
        "Exceptions missed (auto-approved)": int((true_exc & auto).sum()),
        "Exception recall": (true_exc & ~auto).sum() / max(true_exc.sum(), 1),
        "Clean invoices auto-approved": (auto & ~true_exc).sum() / max((~true_exc).sum(), 1),
    }


# ---------------- risk score + queue ----------------
def risk_scores(tables, df: pd.DataFrame) -> pd.Series:
    """Eisen's random forest, trained on the first 80% of history, scores the queue order.
    Its control-flag inputs come from the extracted data, not the ERP."""
    model = ap.build_invoice_model_table(tables)
    train, test = ap.chronological_split(model)
    numeric, categorical = ap.feature_columns()
    features = numeric + categorical
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("preprocess", ap.make_preprocessor(numeric, categorical)),
                     ("model", RandomForestClassifier(n_estimators=350, max_depth=12, min_samples_leaf=4,
                                                      class_weight="balanced_subsample", random_state=42, n_jobs=-1))])
    pipe.fit(train[features], train[ap.TARGET])
    x = test.set_index("Invoice ID").loc[df["invoice_id"]].copy()
    d = df.set_index("invoice_id")
    x["Missing or Invalid PO"] = (d["Missing PO"] | d["Invalid PO"]).values
    for src, dst in [("PO Vendor Mismatch", "PO Vendor Mismatch"), ("Currency Mismatch", "Currency Mismatch"),
                     ("Invoice Over PO", "Invoice Over PO"), ("Payment Terms Mismatch", "Payment Terms Mismatch"),
                     ("Inactive Vendor", "Inactive Vendor"), ("Duplicate Invoice", "Potential Duplicate")]:
        x[dst] = d[src].values
    x["Control Flag Count"] = x[["Missing or Invalid PO", "PO Vendor Mismatch", "Currency Mismatch", "Invoice Over PO",
                                 "Payment Terms Mismatch", "Inactive Vendor", "Potential Duplicate",
                                 "PO Expired at Receipt"]].sum(axis=1)
    return pd.Series(pipe.predict_proba(x[features])[:, 1], index=df["invoice_id"].values)


def queue_sim(tables, df: pd.DataFrame, risk: pd.Series, after_fix: pd.Series | None = None) -> pd.DataFrame:
    """after_fix: the decision each invoice gets once its fields are correct (from the
    perfect-extraction run). A data check that turns up an exception also costs a review."""
    review = tables["Review Log"].copy()
    for c in ["Review Start Timestamp", "Review End Timestamp"]:
        review[c] = pd.to_datetime(review[c])
    svc = ((review["Review End Timestamp"] - review["Review Start Timestamp"]).dt.total_seconds() / 3600).clip(0.25, 8.0)
    svc = pd.Series(svc.values, index=review["Invoice ID"])
    median_svc = float(svc.median())
    fx = tables["FX Rates"].set_index("Currency")["USD per Unit"]
    inv = tables["Invoices"].set_index("Invoice ID")

    q = df[df["Decision"].ne("Auto-approve")].copy()
    # Exception reviews take the observed review time (median if the invoice was never
    # reviewed in the source data). A data check takes 6 minutes, plus the observed
    # review time if the invoice also had a real exception.
    after_fix = after_fix if after_fix is not None else pd.Series(dtype=str)
    q["Service Hours"] = [
        svc.get(i, median_svc) if dec == "Exception review"
        else DATA_CHECK_HOURS + (svc.get(i, median_svc) if after_fix.get(i) == "Exception review" else 0.0)
        for i, dec in zip(q["invoice_id"], q["Decision"])]
    q["Invoice ID"] = q["invoice_id"]
    q["Review Risk Probability"] = q["invoice_id"].map(risk)
    q["Risk Band"] = pd.cut(q["Review Risk Probability"], [-np.inf, 0.35, 0.70, np.inf], labels=["Low", "Medium", "High"]).astype(str)
    q["Invoice Amount USD"] = q["invoice_id"].map(inv["Invoice Amount"] * inv["Currency"].map(fx))
    origin = q["received"].min().normalize()
    origin -= pd.Timedelta(days=origin.weekday())
    q["Arrival Business Hour"] = q["received"].map(lambda t: ap.business_hour_index(t, origin))
    frames = [ap.simulate_queue(q, n, s) for n in range(1, 5) for s in ["FIFO", "Risk Priority"]]
    detail = pd.concat(frames, ignore_index=True)
    summary = detail.groupby(["Reviewer Count", "Queue Strategy"], as_index=False).agg(
        Queue_Items=("Invoice ID", "count"), Reviewer_Hours=("Service Hours", "sum"),
        Avg_Wait_Hours=("Wait Hours", "mean"), SLA_Rate=("Met 8 Business Hour SLA", "mean"))
    return summary


def main() -> None:
    tables, truth, vendors, edi = load_inputs()
    methods = {"Perfect extraction": perfect_extraction(truth)}
    if (OUTPUT / "extracted_rules.csv").exists():
        methods["Rules + OCR"] = pd.read_csv(OUTPUT / "extracted_rules.csv", dtype={"po_number": str, "invoice_number": str})
    if (OUTPUT / "extracted_llm.csv").exists():
        methods["Claude (LLM)"] = pd.read_csv(OUTPUT / "extracted_llm.csv", dtype={"po_number": str, "invoice_number": str})

    edi_df = edi_records(edi)
    weeks = (truth["invoice_id"].map(tables["Invoices"].set_index("Invoice ID")["Received Date"]).agg(["min", "max"]).diff().iloc[-1].days) / 7
    results, acc_rows, decisions, queues = {}, [], [], []
    for name, ext in methods.items():
        full = pd.concat([ext, edi_df], ignore_index=True)
        df = run_controls(full, tables, vendors, truth)
        acc = field_accuracy(df, truth)
        m = decision_metrics(df, tables)
        pdf_count = int((df["text_source"] != "edi").sum())
        manual_pdf = int(((df["text_source"] != "edi") & df["Decision"].ne("Auto-approve")).sum())
        m["Field accuracy (all fields right, PDFs)"] = acc["all_fields"].mean()
        m["Keying hours avoided"] = (pdf_count - manual_pdf) * KEY_MINUTES / 60
        risk = risk_scores(tables, df)
        if name == "Perfect extraction":
            after_fix = df.set_index("invoice_id")["Decision"]
        qs = queue_sim(tables, df, risk, after_fix)
        row = qs[(qs["Reviewer Count"] == REVIEWERS) & (qs["Queue Strategy"] == "Risk Priority")].iloc[0]
        m[f"SLA rate ({REVIEWERS} reviewers, risk priority)"] = row["SLA_Rate"]
        m["Reviewer hours in queue"] = row["Reviewer_Hours"]
        m["Total AP hours (keying + review)"] = row["Reviewer_Hours"]
        m["Weeks covered"] = weeks
        results[name] = m
        for (grp, fmt), g in acc.groupby(["layout_group", "format"]):
            acc_rows.append({"Method": name, "Layout": grp, "Format": fmt, "Invoices": len(g),
                             **{f: g[f].mean() for f in FIELDS + ["all_fields"]}})
        acc_rows.append({"Method": name, "Layout": "All", "Format": "All", "Invoices": len(acc),
                         **{f: acc[f].mean() for f in FIELDS + ["all_fields"]}})
        df["Method"] = name
        df["Review Risk Probability"] = df["invoice_id"].map(risk)
        decisions.append(df)
        qs["Method"] = name
        queues.append(qs)

    # Today's process for comparison: every PDF is keyed by hand, and only the invoices
    # that truly had an exception reach the review queue (the AP project's setup).
    base = decisions[0].copy()
    inv = tables["Invoices"].set_index("Invoice ID")
    true_exc = base["invoice_id"].map(inv["Manual Review Required"]).eq("Yes")
    base["Decision"] = np.where(true_exc, "Exception review", "Manual approval")  # data shows every invoice waits ~3+ days for a person
    qs = queue_sim(tables, base.assign(Decision=np.where(true_exc, "Exception review", "Auto-approve")), base.set_index("invoice_id")["Review Risk Probability"])
    row = qs[(qs["Reviewer Count"] == REVIEWERS) & (qs["Queue Strategy"] == "Risk Priority")].iloc[0]
    lag = base["invoice_id"].map((inv["Approval Date"] - inv["Received Date"]).dt.total_seconds() / 86400)
    results = {"Current process (manual keying)": {
        "Avg days to approve": float(lag.mean()),
        "Invoices": len(base), "Auto-approved (touchless)": int(base["Decision"].eq("Auto-approve").sum()),
        "Touchless rate": base["Decision"].eq("Auto-approve").mean(), "Exception review": int(true_exc.sum()),
        "Data check": 0, "True exceptions": int(true_exc.sum()), "Exceptions caught": int(true_exc.sum()),
        "Exceptions missed (auto-approved)": 0, "Exception recall": 1.0, "Clean invoices auto-approved": np.nan,
        "Field accuracy (all fields right, PDFs)": np.nan, "Keying hours avoided": 0.0,
        f"SLA rate ({REVIEWERS} reviewers, risk priority)": row["SLA_Rate"], "Reviewer hours in queue": row["Reviewer_Hours"],
        "Total AP hours (keying + review)": row["Reviewer_Hours"] + int((base["text_source"] != "edi").sum()) * KEY_MINUTES / 60,
        "Weeks covered": weeks}} | results
    qs["Method"] = "Current process (manual keying)"
    queues.insert(0, qs)
    summary = pd.DataFrame(results).T
    accuracy = pd.DataFrame(acc_rows)
    decisions = pd.concat(decisions, ignore_index=True)
    queues = pd.concat(queues, ignore_index=True)
    summary.to_csv(OUTPUT / "summary_by_method.csv")
    accuracy.to_csv(OUTPUT / "field_accuracy.csv", index=False)
    decisions.to_csv(OUTPUT / "invoice_decisions.csv", index=False)
    queues.to_csv(OUTPUT / "queue_simulation.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(summary.T.to_string())
    print(accuracy.round(3).to_string())


if __name__ == "__main__":
    main()
