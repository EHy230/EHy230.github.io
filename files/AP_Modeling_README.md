# Accounts Payable AI & Finance Operations

An end-to-end finance operations portfolio project combining Excel controls, Python predictive modeling, queue-based staffing analysis, and Tableau data storytelling. The dataset is synthetic and contains 5,000 invoices.

## Portfolio deliverables

- `AP_Finance_Operations_Eisen.xlsx` — rule-based invoice controls, exception analysis, prioritization, KPIs, and management reporting.
- `AP_Finance_Operations_Predictive_Model.ipynb` — self-contained Google Colab notebook covering data preparation, model comparison, threshold selection, validation, and staffing simulation.
- `AP_Finance_Operations_AI_Dashboard_Eisen_Hy_Final.twbx` — packaged Tableau dashboard with an embedded data extract.
- `ap_modeling_pipeline.py` — reproducible command-line version of the Python workflow.
- `requirements.txt` — Python dependencies.

## Business question

How can an accounts payable team identify invoices most likely to require manual review, prioritize high-risk work, and determine the reviewer capacity needed to meet an eight-business-hour service target?

## Approach

1. Tested invoices against duplicate, purchase-order, currency, payment-term, and vendor-master controls in Excel.
2. Used a chronological 80/20 train/test split to avoid learning from future outcomes.
3. Compared a majority baseline, logistic regression, and random forest model.
4. Selected an operating threshold designed to retain at least 85% recall.
5. Simulated one to four reviewers under FIFO and risk-priority queueing policies.
6. Presented operational KPIs and risk patterns in Tableau.

## Key results

- Random forest selected as the final model.
- 97.3% test accuracy.
- 92.2% precision and 97.1% recall for future-period review cases.
- Two reviewers using risk-priority routing achieved a 96.1% simulated eight-business-hour SLA.
- Average high-risk wait time fell to approximately 0.22 business hours.

## Running the notebook

1. Open Google Colab.
2. Upload `AP_Finance_Operations_Predictive_Model.ipynb`.
3. Run the cells from top to bottom.
4. Upload `AP_Finance_Operations_Eisen.xlsx` when prompted.

The Tableau `.twbx` file is portable and can be opened directly in Tableau Public or Tableau Desktop without reconnecting the Excel source.
