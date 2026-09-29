# AP Finance Project

Two connected projects on the same synthetic accounts payable data (5,000 invoices, 140 vendors).

## 1 - AP Modeling
Which invoices need a manual review, and how many reviewers does the AP team need?
- `AP_Finance_Operations_Eisen.xlsx`: source data, invoice controls, exception analysis and KPIs in Excel
- `AP_Finance_Operations_Predictive_Model.ipynb`: Colab notebook with the review-risk model (random forest) and the reviewer staffing simulation
- `ap_modeling_pipeline.py`: the same workflow as a script
- `AP_Finance_Operations_AI_Dashboard_Eisen_Hy_Final.twbx`: Tableau dashboard
- `Tableau_Data/`: the tables the dashboard is built from

## 2 - AI Invoice Processing
The follow-up. It automates the typing step. Claude reads 836 invoice PDFs, checks each one against its purchase order and the vendor master, approves the clean ones automatically, and sends the rest to the review queue from project 1.
- `README.md`: what it does and the results (70% of invoices approved with no one touching them, 97.8% of PDFs read correctly, $3.98 total cost)
- `AI_Invoice_Processing.ipynb`: Colab notebook that runs the Claude step
- `src/`: the Python code
- `invoices/`: the 836 invoice PDFs
- `output/`: results tables (usable in Tableau)
- `docs/sample_invoices.png`: sample invoices for the portfolio
