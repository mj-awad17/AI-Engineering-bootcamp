# Sales Data Notebook

The notebook explores sales-data quality, summarizes sales, and practices editing CSV files. Its work is organized into two parts:

## Part A — Data Quality & Basic Insights

- How can the raw sales data be loaded and inspected?
- How should inconsistent text, missing values, invalid prices, and duplicate rows be handled while preserving the original data?
- How does revenue get calculated, and how can revenue be summarized by region or category?
- What do two labelled charts show, and what evidence-based findings can be drawn from them?
- How can the cleaned data be saved as `cleaned_sales.csv`?

## Part B — CSV File Handling & Manipulation

- How can `cleaned_sales.csv` be loaded, and a 5% tax amount added to revenue?
- How does a 10% Books price increase affect revenue and tax?
- How can a selected transaction be removed and a temporary review-status column added and then removed?
- How can the updated data be saved as `sales_updated.csv` and verified?

## Requirements

- Python 3
- Jupyter Notebook
- pandas and NumPy

Install the requirements if needed:

```bash
python -m pip install notebook pandas numpy
```

## Run the notebook

1. Open a terminal in the `Week-2/Assignment` folder.
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `mjawad_003721.ipynb`.
4. Choose **Kernel → Restart & Run All** (or **Run All**) to execute the cells in order.

Keep `sales_raw.csv` in the same folder as the notebook; the notebook loads it by filename.

The notebook reads `sales_raw.csv` and is intended to save `cleaned_sales.csv` and `sales_updated.csv` in the same folder.