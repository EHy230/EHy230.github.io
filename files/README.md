# AP Finance Project files

Files for my accounts payable project, which has two connected parts built on the same synthetic data (5,000 invoices, 140 vendors). This folder also holds files for my other portfolio projects.

## Part 1: AP modeling (Excel, Python, Tableau)
Which invoices need a manual review, and how many reviewers does the AP team need? See [AP_Modeling_README.md](AP_Modeling_README.md).
- `AP_Finance_Operations_Eisen.xlsx`: source data, invoice checks, priority scoring, payment and review analysis
- `AP_Finance_Operations_Predictive_Model.ipynb`: Colab notebook with the review-risk model (random forest) and the reviewer staffing simulation
- `ap_modeling_pipeline.py`: the same workflow as a script
- `AP_Finance_Operations_AI_Dashboard_Eisen_Hy_Final.twbx`: Tableau dashboard
- `AP_Tableau_Outputs.xlsx`, `tableau_*.csv`: the tables the dashboard is built from

## Part 2: AI invoice processing (Claude API)
Claude reads 836 invoice PDFs and checks each one against its purchase order and the vendor master. Clean invoices are approved automatically, and the rest go to the review queue from Part 1. See [AI_Invoice_Processing_README.md](AI_Invoice_Processing_README.md).
- `AI_Invoice_Processing.ipynb`: Colab notebook that runs the Claude step
- `common.py`, `generate_invoices.py`, `extract_rules.py`, `extract_llm.py`, `process.py`: the Python code
- `summary_by_method.csv`, `field_accuracy.csv`, `invoice_decisions.csv`, `queue_simulation.csv`: results
- `extracted_llm.csv`, `extracted_rules.csv`, `ground_truth.csv`, `edi_feed.json`, `vendor_master_addresses.csv`: extraction outputs and inputs
- `sample_invoices.png`: sample invoices
