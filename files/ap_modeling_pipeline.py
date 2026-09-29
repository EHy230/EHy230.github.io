"""Accounts Payable review-risk model and Tableau export pipeline.

This module is intentionally usable both from a Google Colab notebook and as a
command-line script.  It reads the polished AP workbook, engineers only fields
that would be available when an invoice enters Accounts Payable, evaluates two
interpretable/predictive models on a chronological holdout, refits the selected
model on all records, runs a deterministic staffing simulation, and creates
Tableau-ready files.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET = "Manual Review Flag"


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    probabilities: np.ndarray
    metrics: dict[str, float | int | str]


def load_source_tables(workbook_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load the six source tables used by the project."""
    workbook_path = Path(workbook_path)
    required = [
        "Invoices",
        "Vendors",
        "Purchase Orders",
        "FX Rates",
        "Payments",
        "Review Log",
    ]
    tables = pd.read_excel(workbook_path, sheet_name=required)
    missing = [name for name in required if name not in tables]
    if missing:
        raise ValueError(f"Missing required sheets: {missing}")
    return tables


def _yes_no_to_int(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().eq("yes").astype(int)


def build_invoice_model_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create the invoice-level feature table without post-review leakage."""
    invoices = tables["Invoices"].copy()
    vendors = tables["Vendors"].copy()
    purchase_orders = tables["Purchase Orders"].copy()
    fx = tables["FX Rates"].copy()

    date_columns = ["Invoice Date", "Received Date", "Due Date", "Approval Date"]
    for column in date_columns:
        invoices[column] = pd.to_datetime(invoices[column], errors="coerce")

    vendor_columns = [
        "Vendor ID",
        "Vendor Name",
        "Category",
        "Country",
        "Default Currency",
        "Default Terms Days",
        "Risk Tier",
        "Preferred Vendor",
        "Active Flag",
    ]
    vendors = vendors[vendor_columns].rename(
        columns={
            "Category": "Vendor Category",
            "Country": "Vendor Country",
            "Active Flag": "Vendor Active Flag",
        }
    )

    po_columns = [
        "PO ID",
        "Vendor ID",
        "PO Amount",
        "Currency",
        "Approval Status",
        "Valid Through",
    ]
    purchase_orders = purchase_orders[po_columns].rename(
        columns={
            "Vendor ID": "PO Vendor ID",
            "PO Amount": "PO Amount",
            "Currency": "PO Currency",
            "Approval Status": "PO Approval Status",
            "Valid Through": "PO Valid Through",
        }
    )
    purchase_orders["PO Valid Through"] = pd.to_datetime(
        purchase_orders["PO Valid Through"], errors="coerce"
    )

    fx_lookup = fx.set_index("Currency")["USD per Unit"]
    model = invoices.merge(vendors, on="Vendor ID", how="left", validate="many_to_one")
    model = model.merge(purchase_orders, on="PO ID", how="left", validate="many_to_one")

    model["FX Rate"] = model["Currency"].map(fx_lookup)
    if model["FX Rate"].isna().any():
        currencies = sorted(model.loc[model["FX Rate"].isna(), "Currency"].dropna().unique())
        raise ValueError(f"Missing FX rates for currencies: {currencies}")
    model["Invoice Amount USD"] = model["Invoice Amount"] * model["FX Rate"]
    model["Log Invoice Amount USD"] = np.log1p(model["Invoice Amount USD"].clip(lower=0))

    model["Missing or Invalid PO"] = (
        model["PO ID"].isna() | model["PO Amount"].isna()
    ).astype(int)
    model["PO Vendor Mismatch"] = (
        model["PO Amount"].notna()
        & model["PO Vendor ID"].notna()
        & model["Vendor ID"].ne(model["PO Vendor ID"])
    ).astype(int)
    model["Currency Mismatch"] = (
        model["PO Amount"].notna()
        & model["PO Currency"].notna()
        & model["Currency"].ne(model["PO Currency"])
    ).astype(int)
    # Do not compare numeric invoice and PO amounts unless their currencies match.
    model["Invoice Over PO"] = (
        model["PO Amount"].notna()
        & model["Currency"].eq(model["PO Currency"])
        & model["Invoice Amount"].gt(model["PO Amount"])
    ).astype(int)
    model["Payment Terms Mismatch"] = (
        model["Default Terms Days"].notna()
        & model["Payment Terms Days"].ne(model["Default Terms Days"])
    ).astype(int)
    model["Inactive Vendor"] = (
        model["Vendor Active Flag"].astype("string").str.strip().str.lower().ne("yes")
    ).astype(int)
    duplicate_key = ["Vendor ID", "Invoice Number"]
    model["Potential Duplicate"] = model.duplicated(duplicate_key, keep=False).astype(int)
    model["PO Expired at Receipt"] = (
        model["PO Valid Through"].notna()
        & model["Received Date"].gt(model["PO Valid Through"])
    ).astype(int)
    model["Days Invoice to Receipt"] = (
        model["Received Date"] - model["Invoice Date"]
    ).dt.total_seconds().div(86400).clip(lower=0)
    model["Received Month"] = model["Received Date"].dt.month.astype("Int64")
    model["Received Weekday"] = model["Received Date"].dt.day_name()
    model["Received Quarter"] = "Q" + model["Received Date"].dt.quarter.astype("Int64").astype("string")
    model["Manual Review Flag"] = _yes_no_to_int(model["Manual Review Required"])

    control_columns = [
        "Missing or Invalid PO",
        "PO Vendor Mismatch",
        "Currency Mismatch",
        "Invoice Over PO",
        "Payment Terms Mismatch",
        "Inactive Vendor",
        "Potential Duplicate",
        "PO Expired at Receipt",
    ]
    model["Control Flag Count"] = model[control_columns].sum(axis=1)
    return model.sort_values(["Received Date", "Invoice ID"]).reset_index(drop=True)


def feature_columns() -> tuple[list[str], list[str]]:
    numeric = [
        "Invoice Amount USD",
        "Log Invoice Amount USD",
        "Payment Terms Days",
        "Days Invoice to Receipt",
        "Received Month",
        "Missing or Invalid PO",
        "PO Vendor Mismatch",
        "Currency Mismatch",
        "Invoice Over PO",
        "Payment Terms Mismatch",
        "Inactive Vendor",
        "Potential Duplicate",
        "PO Expired at Receipt",
        "Control Flag Count",
    ]
    categorical = [
        "Department",
        "Cost Center",
        "Invoice Type",
        "Submitted Channel",
        "Currency",
        "Vendor Category",
        "Vendor Country",
        "Risk Tier",
        "Preferred Vendor",
        "PO Approval Status",
        "Received Weekday",
        "Received Quarter",
    ]
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def chronological_split(
    model_table: pd.DataFrame, test_fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(math.floor(len(model_table) * (1 - test_fraction)))
    if split_index <= 0 or split_index >= len(model_table):
        raise ValueError("The chronological split produced an empty train or test set.")
    return model_table.iloc[:split_index].copy(), model_table.iloc[split_index:].copy()


def _evaluate(
    name: str,
    pipeline: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> ModelResult:
    pipeline.fit(train[features], train[TARGET])
    probabilities = pipeline.predict_proba(test[features])[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(test[TARGET], predictions, labels=[0, 1]).ravel()
    metrics: dict[str, float | int | str] = {
        "Model": name,
        "Train Rows": len(train),
        "Test Rows": len(test),
        "Test Start": test["Received Date"].min().date().isoformat(),
        "Test End": test["Received Date"].max().date().isoformat(),
        "ROC AUC": roc_auc_score(test[TARGET], probabilities),
        "Average Precision": average_precision_score(test[TARGET], probabilities),
        "Accuracy": accuracy_score(test[TARGET], predictions),
        "Precision": precision_score(test[TARGET], predictions, zero_division=0),
        "Recall": recall_score(test[TARGET], predictions, zero_division=0),
        "F1": f1_score(test[TARGET], predictions, zero_division=0),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
    }
    return ModelResult(name, pipeline, probabilities, metrics)


def train_and_compare_models(
    model_table: pd.DataFrame,
) -> tuple[ModelResult, list[ModelResult], pd.DataFrame, pd.DataFrame, list[str]]:
    numeric, categorical = feature_columns()
    features = numeric + categorical
    train, test = chronological_split(model_table)

    models: list[tuple[str, object]] = [
        ("Majority Baseline", DummyClassifier(strategy="prior")),
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=350,
                max_depth=12,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]

    results: list[ModelResult] = []
    for name, estimator in models:
        pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(numeric, categorical)),
                ("model", estimator),
            ]
        )
        results.append(_evaluate(name, pipeline, train, test, features))

    candidates = [result for result in results if result.name != "Majority Baseline"]
    selected = max(
        candidates,
        key=lambda result: (float(result.metrics["Average Precision"]), float(result.metrics["Recall"])),
    )
    return selected, results, train, test, features


def select_operating_threshold(y_true: Iterable[int], probabilities: np.ndarray) -> float:
    """Pick the most precise threshold that retains at least 85% recall."""
    y = np.asarray(list(y_true), dtype=int)
    thresholds = np.linspace(0.05, 0.95, 181)
    candidates: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        recall = recall_score(y, pred, zero_division=0)
        precision = precision_score(y, pred, zero_division=0)
        if recall >= 0.85:
            candidates.append((precision, threshold, recall))
    if not candidates:
        return 0.50
    return float(max(candidates, key=lambda item: (item[0], item[1]))[1])


def feature_importance_table(pipeline: Pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        direction = np.repeat("Importance", len(values))
    elif hasattr(model, "coef_"):
        signed = np.asarray(model.coef_[0])
        values = np.abs(signed)
        direction = np.where(signed >= 0, "Raises review risk", "Lowers review risk")
    else:
        raise TypeError("The selected model does not expose feature importance.")
    output = pd.DataFrame(
        {"Feature": names, "Importance": values, "Direction": direction}
    ).sort_values("Importance", ascending=False)
    output["Rank"] = np.arange(1, len(output) + 1)
    return output[["Rank", "Feature", "Importance", "Direction"]].reset_index(drop=True)


def score_all_invoices(
    model_table: pd.DataFrame,
    selected: ModelResult,
    features: list[str],
    test: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame, float]:
    threshold = select_operating_threshold(test[TARGET], selected.probabilities)
    selected.pipeline.fit(model_table[features], model_table[TARGET])
    scored = model_table.copy()
    scored["Review Risk Probability"] = selected.pipeline.predict_proba(scored[features])[:, 1]
    scored["Predicted Review Flag"] = (
        scored["Review Risk Probability"] >= threshold
    ).astype(int)
    scored["Risk Band"] = pd.cut(
        scored["Review Risk Probability"],
        bins=[-np.inf, 0.35, 0.70, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype("string")
    test_ids = set(test["Invoice ID"])
    scored["Evaluation Segment"] = np.where(
        scored["Invoice ID"].isin(test_ids), "Chronological Test", "Training Period"
    )
    return selected.pipeline, scored, threshold


def _next_business_day(day: pd.Timestamp) -> pd.Timestamp:
    day = day.normalize()
    while day.weekday() >= 5:
        day += pd.Timedelta(days=1)
    return day


def business_hour_index(timestamp: pd.Timestamp, origin: pd.Timestamp) -> float:
    """Map a timestamp to elapsed Mon-Fri, 9:00-17:00 business hours."""
    timestamp = pd.Timestamp(timestamp)
    day = timestamp.normalize()
    if day.weekday() >= 5:
        day = _next_business_day(day)
        offset = 0.0
    else:
        hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
        if hour < 9:
            offset = 0.0
        elif hour >= 17:
            day = _next_business_day(day + pd.Timedelta(days=1))
            offset = 0.0
        else:
            offset = hour - 9.0
    business_days = np.busday_count(origin.date(), day.date())
    return float(business_days * 8 + offset)


def simulate_queue(
    queue: pd.DataFrame,
    reviewer_count: int,
    strategy: str,
) -> pd.DataFrame:
    """Deterministic multi-reviewer queue using observed arrivals/service times."""
    arrivals = queue.sort_values(["Arrival Business Hour", "Invoice ID"]).reset_index(drop=True)
    next_arrival = 0
    pending: list[int] = []
    servers: list[tuple[float, int]] = [(0.0, server) for server in range(reviewer_count)]
    heapq.heapify(servers)
    records: list[dict[str, float | int | str]] = []

    while next_arrival < len(arrivals) or pending:
        available, server_id = heapq.heappop(servers)
        while (
            next_arrival < len(arrivals)
            and arrivals.loc[next_arrival, "Arrival Business Hour"] <= available + 1e-12
        ):
            pending.append(next_arrival)
            next_arrival += 1

        if not pending and next_arrival < len(arrivals):
            available = float(arrivals.loc[next_arrival, "Arrival Business Hour"])
            while (
                next_arrival < len(arrivals)
                and arrivals.loc[next_arrival, "Arrival Business Hour"] <= available + 1e-12
            ):
                pending.append(next_arrival)
                next_arrival += 1

        if not pending:
            heapq.heappush(servers, (available, server_id))
            continue

        if strategy == "FIFO":
            chosen_position = min(
                range(len(pending)),
                key=lambda pos: (
                    arrivals.loc[pending[pos], "Arrival Business Hour"],
                    pending[pos],
                ),
            )
        elif strategy == "Risk Priority":
            chosen_position = max(
                range(len(pending)),
                key=lambda pos: (
                    arrivals.loc[pending[pos], "Review Risk Probability"],
                    arrivals.loc[pending[pos], "Invoice Amount USD"],
                    -arrivals.loc[pending[pos], "Arrival Business Hour"],
                ),
            )
        else:
            raise ValueError(f"Unknown queue strategy: {strategy}")

        row_number = pending.pop(chosen_position)
        row = arrivals.loc[row_number]
        start = max(available, float(row["Arrival Business Hour"]))
        end = start + float(row["Service Hours"])
        wait = start - float(row["Arrival Business Hour"])
        turnaround = end - float(row["Arrival Business Hour"])
        records.append(
            {
                "Invoice ID": row["Invoice ID"],
                "Reviewer Count": reviewer_count,
                "Queue Strategy": strategy,
                "Simulated Reviewer": f"Reviewer {server_id + 1}",
                "Review Risk Probability": row["Review Risk Probability"],
                "Risk Band": row["Risk Band"],
                "Invoice Amount USD": row["Invoice Amount USD"],
                "Arrival Business Hour": row["Arrival Business Hour"],
                "Simulated Start Business Hour": start,
                "Simulated End Business Hour": end,
                "Service Hours": row["Service Hours"],
                "Wait Hours": wait,
                "Turnaround Hours": turnaround,
                "Met 8 Business Hour SLA": int(turnaround <= 8.0),
            }
        )
        heapq.heappush(servers, (end, server_id))

    return pd.DataFrame(records)


def build_queue_outputs(
    tables: dict[str, pd.DataFrame], scored: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    review = tables["Review Log"].copy()
    for column in ["Queue Entry Timestamp", "Review Start Timestamp", "Review End Timestamp"]:
        review[column] = pd.to_datetime(review[column], errors="coerce")
    review["Service Hours"] = (
        review["Review End Timestamp"] - review["Review Start Timestamp"]
    ).dt.total_seconds().div(3600).clip(lower=0.25, upper=8.0)
    queue = review.merge(
        scored[
            [
                "Invoice ID",
                "Invoice Amount USD",
                "Review Risk Probability",
                "Risk Band",
            ]
        ],
        on="Invoice ID",
        how="left",
        validate="one_to_one",
    ).dropna(subset=["Queue Entry Timestamp", "Service Hours"])

    origin = queue["Queue Entry Timestamp"].min().normalize()
    origin -= pd.Timedelta(days=origin.weekday())
    queue["Arrival Business Hour"] = queue["Queue Entry Timestamp"].map(
        lambda value: business_hour_index(value, origin)
    )

    detail_frames = [
        simulate_queue(queue, reviewer_count, strategy)
        for reviewer_count in range(1, 5)
        for strategy in ["FIFO", "Risk Priority"]
    ]
    detail = pd.concat(detail_frames, ignore_index=True)
    summary = (
        detail.groupby(["Reviewer Count", "Queue Strategy"], as_index=False)
        .agg(
            Reviews=("Invoice ID", "count"),
            Average_Wait_Hours=("Wait Hours", "mean"),
            Median_Wait_Hours=("Wait Hours", "median"),
            P90_Wait_Hours=("Wait Hours", lambda value: value.quantile(0.90)),
            Average_Turnaround_Hours=("Turnaround Hours", "mean"),
            SLA_Rate=("Met 8 Business Hour SLA", "mean"),
        )
        .rename(
            columns={
                "Average_Wait_Hours": "Average Wait Hours",
                "Median_Wait_Hours": "Median Wait Hours",
                "P90_Wait_Hours": "P90 Wait Hours",
                "Average_Turnaround_Hours": "Average Turnaround Hours",
                "SLA_Rate": "8 Business Hour SLA Rate",
            }
        )
    )
    high_risk = (
        detail.loc[detail["Risk Band"].eq("High")]
        .groupby(["Reviewer Count", "Queue Strategy"])["Wait Hours"]
        .mean()
        .rename("High Risk Average Wait Hours")
        .reset_index()
    )
    summary = summary.merge(high_risk, on=["Reviewer Count", "Queue Strategy"], how="left")
    return summary, detail


def build_tableau_exports(
    tables: dict[str, pd.DataFrame],
    scored: pd.DataFrame,
    model_metrics: pd.DataFrame,
    importance: pd.DataFrame,
    queue_summary: pd.DataFrame,
    queue_detail: pd.DataFrame,
    threshold: float,
    selected_model_name: str,
) -> dict[str, pd.DataFrame]:
    payments = tables["Payments"].copy()
    payments["Late Payment Flag"] = (payments["Days From Due"] > 0).astype(int)
    payment_invoice = payments.groupby("Invoice ID", as_index=False).agg(
        Payment_Status=("Payment Status", "first"),
        Days_From_Due=("Days From Due", "max"),
        Late_Payment_Flag=("Late Payment Flag", "max"),
        Discount_Amount=("Discount Amount", "sum"),
        Late_Fee=("Late Fee", "sum"),
    ).rename(
        columns={
            "Payment_Status": "Payment Status",
            "Days_From_Due": "Days From Due",
            "Late_Payment_Flag": "Late Payment Flag",
            "Discount_Amount": "Discount Amount",
            "Late_Fee": "Late Fee",
        }
    )

    review = tables["Review Log"].copy()
    review["Review Start Timestamp"] = pd.to_datetime(review["Review Start Timestamp"], errors="coerce")
    review["Review End Timestamp"] = pd.to_datetime(review["Review End Timestamp"], errors="coerce")
    review["Review Turnaround Hours"] = (
        review["Review End Timestamp"] - pd.to_datetime(review["Queue Entry Timestamp"], errors="coerce")
    ).dt.total_seconds().div(3600)
    review_invoice = review[
        [
            "Invoice ID",
            "Reviewer ID",
            "Resolution",
            "Resolution Code",
            "Rework Flag",
            "Review Turnaround Hours",
        ]
    ]

    export_columns = [
        "Invoice ID",
        "Invoice Number",
        "Vendor ID",
        "Vendor Name",
        "Vendor Category",
        "Vendor Country",
        "Risk Tier",
        "Preferred Vendor",
        "Department",
        "Cost Center",
        "Invoice Type",
        "Submitted Channel",
        "Currency",
        "Invoice Date",
        "Received Date",
        "Due Date",
        "Invoice Amount USD",
        "Manual Review Flag",
        "Exception Type",
        "Control Flag Count",
        "Missing or Invalid PO",
        "PO Vendor Mismatch",
        "Currency Mismatch",
        "Invoice Over PO",
        "Payment Terms Mismatch",
        "Inactive Vendor",
        "Potential Duplicate",
        "PO Expired at Receipt",
        "Review Risk Probability",
        "Predicted Review Flag",
        "Risk Band",
        "Evaluation Segment",
    ]
    invoice_predictions = scored[export_columns].copy()
    invoice_predictions = invoice_predictions.merge(
        payment_invoice, on="Invoice ID", how="left", validate="one_to_one"
    ).merge(review_invoice, on="Invoice ID", how="left", validate="one_to_one")
    invoice_predictions["Manual Review Outcome"] = np.where(
        invoice_predictions["Manual Review Flag"].eq(1), "Reviewed", "No Review"
    )
    invoice_predictions["Prediction Outcome"] = np.select(
        [
            invoice_predictions["Manual Review Flag"].eq(1)
            & invoice_predictions["Predicted Review Flag"].eq(1),
            invoice_predictions["Manual Review Flag"].eq(0)
            & invoice_predictions["Predicted Review Flag"].eq(0),
            invoice_predictions["Manual Review Flag"].eq(0)
            & invoice_predictions["Predicted Review Flag"].eq(1),
        ],
        ["True Positive", "True Negative", "False Positive"],
        default="False Negative",
    )
    invoice_predictions["Received Date Only"] = invoice_predictions["Received Date"].dt.date

    daily = (
        invoice_predictions.groupby("Received Date Only", as_index=False)
        .agg(
            Invoice_Count=("Invoice ID", "count"),
            Invoice_Value_USD=("Invoice Amount USD", "sum"),
            Actual_Reviews=("Manual Review Flag", "sum"),
            Predicted_Reviews=("Predicted Review Flag", "sum"),
            Average_Risk=("Review Risk Probability", "mean"),
            High_Risk_Invoices=("Risk Band", lambda value: value.eq("High").sum()),
            Late_Payments=("Late Payment Flag", "sum"),
            Late_Fees=("Late Fee", "sum"),
        )
        .rename(
            columns={
                "Received Date Only": "Date",
                "Invoice_Count": "Invoice Count",
                "Invoice_Value_USD": "Invoice Value USD",
                "Actual_Reviews": "Actual Reviews",
                "Predicted_Reviews": "Predicted Reviews",
                "Average_Risk": "Average Review Risk",
                "High_Risk_Invoices": "High Risk Invoices",
                "Late_Payments": "Late Payments",
                "Late_Fees": "Late Fees USD",
            }
        )
    )

    vendor = (
        invoice_predictions.groupby(
            ["Vendor ID", "Vendor Name", "Vendor Category", "Risk Tier", "Preferred Vendor"],
            dropna=False,
            as_index=False,
        )
        .agg(
            Invoice_Count=("Invoice ID", "count"),
            Invoice_Value_USD=("Invoice Amount USD", "sum"),
            Review_Count=("Manual Review Flag", "sum"),
            Average_Risk=("Review Risk Probability", "mean"),
            High_Risk_Invoices=("Risk Band", lambda value: value.eq("High").sum()),
            Late_Payments=("Late Payment Flag", "sum"),
            Rework_Count=("Rework Flag", lambda value: value.astype("string").str.lower().eq("yes").sum()),
        )
        .rename(
            columns={
                "Invoice_Count": "Invoice Count",
                "Invoice_Value_USD": "Invoice Value USD",
                "Review_Count": "Review Count",
                "Average_Risk": "Average Review Risk",
                "High_Risk_Invoices": "High Risk Invoices",
                "Late_Payments": "Late Payments",
                "Rework_Count": "Rework Count",
            }
        )
    )
    vendor["Review Rate"] = vendor["Review Count"] / vendor["Invoice Count"]
    vendor["Late Payment Rate"] = vendor["Late Payments"] / vendor["Invoice Count"]

    # Recommend the smallest staffing level that reaches a practical 95% SLA;
    # within that staffing level, prefer the stronger SLA and lower wait time.
    eligible_queue = queue_summary.loc[queue_summary["8 Business Hour SLA Rate"] >= 0.95]
    if eligible_queue.empty:
        eligible_queue = queue_summary
    minimum_reviewers = eligible_queue["Reviewer Count"].min()
    best_queue = (
        eligible_queue.loc[eligible_queue["Reviewer Count"].eq(minimum_reviewers)]
        .sort_values(
            ["8 Business Hour SLA Rate", "Average Wait Hours"],
            ascending=[False, True],
        )
        .iloc[0]
    )
    kpis = pd.DataFrame(
        [
            {
                "Selected Model": selected_model_name,
                "Operating Threshold": threshold,
                "Total Invoices": len(invoice_predictions),
                "Invoice Value USD": invoice_predictions["Invoice Amount USD"].sum(),
                "Actual Review Rate": invoice_predictions["Manual Review Flag"].mean(),
                "Predicted Review Rate": invoice_predictions["Predicted Review Flag"].mean(),
                "High Risk Invoices": invoice_predictions["Risk Band"].eq("High").sum(),
                "Late Payment Rate": invoice_predictions["Late Payment Flag"].fillna(0).mean(),
                "Recommended Starting Reviewers": int(best_queue["Reviewer Count"]),
                "Recommended Queue Strategy": best_queue["Queue Strategy"],
                "Queue SLA Definition": "Completed within 8 business hours",
            }
        ]
    )

    return {
        "Invoice Predictions": invoice_predictions,
        "Daily Operations": daily,
        "Vendor Summary": vendor,
        "Model Metrics": model_metrics,
        "Feature Importance": importance,
        "Queue Scenarios": queue_summary,
        "Queue Detail": queue_detail,
        "Dashboard KPIs": kpis,
    }


def export_outputs(
    exports: dict[str, pd.DataFrame],
    output_dir: str | Path,
    pipeline: Pipeline,
    metadata: dict,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_names = {
        "Invoice Predictions": "tableau_invoice_predictions.csv",
        "Daily Operations": "tableau_daily_operations.csv",
        "Vendor Summary": "tableau_vendor_summary.csv",
        "Model Metrics": "tableau_model_metrics.csv",
        "Feature Importance": "tableau_feature_importance.csv",
        "Queue Scenarios": "tableau_queue_scenarios.csv",
        "Queue Detail": "tableau_queue_detail.csv",
        "Dashboard KPIs": "tableau_dashboard_kpis.csv",
    }
    for key, file_name in csv_names.items():
        exports[key].to_csv(output_dir / file_name, index=False)

    tableau_workbook = output_dir / "AP_Tableau_Outputs.xlsx"
    with pd.ExcelWriter(tableau_workbook, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm") as writer:
        for sheet_name, frame in exports.items():
            safe_name = sheet_name[:31]
            frame.to_excel(writer, sheet_name=safe_name, index=False)
            worksheet = writer.sheets[safe_name]
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
            worksheet.set_row(0, 22)
            for index, column in enumerate(frame.columns):
                width = min(max(len(str(column)) + 2, 12), 34)
                worksheet.set_column(index, index, width)

    joblib.dump(pipeline, output_dir / "ap_review_risk_model.joblib")
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    # Keep the archive outside its source directory so it cannot recursively
    # include itself while it is being written.
    archive_base = output_dir.parent / "AP_Tableau_Package"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", output_dir))
    return archive_path


def run_pipeline(workbook_path: str | Path, output_dir: str | Path) -> dict:
    tables = load_source_tables(workbook_path)
    model_table = build_invoice_model_table(tables)
    selected, results, train, test, features = train_and_compare_models(model_table)
    operating_model, scored, threshold = score_all_invoices(
        model_table, selected, features, test
    )
    model_metrics = pd.DataFrame([result.metrics for result in results])
    model_metrics["Selected Model"] = model_metrics["Model"].eq(selected.name)
    importance = feature_importance_table(operating_model)
    queue_summary, queue_detail = build_queue_outputs(tables, scored)
    exports = build_tableau_exports(
        tables,
        scored,
        model_metrics,
        importance,
        queue_summary,
        queue_detail,
        threshold,
        selected.name,
    )
    metadata = {
        "input_workbook": str(workbook_path),
        "selected_model": selected.name,
        "operating_threshold": threshold,
        "training_rows": len(train),
        "test_rows": len(test),
        "test_start": test["Received Date"].min(),
        "test_end": test["Received Date"].max(),
        "features": features,
        "target": TARGET,
        "leakage_controls": [
            "Chronological 80/20 split",
            "No exception type in predictive features",
            "No review log outcomes in predictive features",
            "No payment outcomes in predictive features",
        ],
        "queue_assumptions": [
            "Mon-Fri, 9:00-17:00 operating window",
            "Observed review handling time, clipped to 0.25-8.0 hours",
            "8-business-hour completion SLA",
            "Deterministic scenario analysis; not Monte Carlo",
        ],
    }
    archive = export_outputs(exports, output_dir, operating_model, metadata)
    return {
        "selected_model": selected.name,
        "operating_threshold": threshold,
        "metrics": model_metrics,
        "feature_importance": importance,
        "queue_summary": queue_summary,
        "exports": exports,
        "archive": archive,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to AP Excel workbook")
    parser.add_argument("--output", required=True, help="Directory for model/Tableau outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(args.input, args.output)
    print(f"Selected model: {result['selected_model']}")
    print(f"Operating threshold: {result['operating_threshold']:.3f}")
    print("\nModel metrics:")
    print(result["metrics"].to_string(index=False))
    print("\nQueue scenarios:")
    print(result["queue_summary"].to_string(index=False))
    print(f"\nCreated Tableau package: {result['archive']}")
