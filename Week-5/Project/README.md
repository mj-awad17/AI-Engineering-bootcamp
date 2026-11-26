# AI-Powered Ticket Classification Assistant

A local customer-support ticket classifier that compares calibrated TF-IDF machine-learning predictions with Google Gemini and explains its hybrid decision. The API and Streamlit interface run from this project folder.

## Features

- Ten fixed ticket categories, shared and validated across the API, ML pipeline, and Gemini output.
- Validation-set comparison of Logistic Regression, calibrated Linear SVM, Multinomial Naive Bayes, and Random Forest; the highest Macro F1 model is persisted, with a separate Random Forest artifact for inference comparison.
- Gemini zero-shot, instruction-based, and few-shot prompt styles with strict structured-response validation.
- A confidence- and agreement-based hybrid decision with explicit human-review flags.
- FastAPI endpoints, interactive Streamlit UI, evaluation metrics, and a confusion-matrix plot.
- A small, project-authored synthetic CSV for exercising the workflow without distributing third-party data.

## Categories

Technical Support; Billing & Payments; Account & Access; Product Support; Sales & Pre-Sales; Returns & Refunds; Customer Service; IT & Infrastructure; Service Outages; General Inquiries.

## Model walkthrough notebook

Open [notebook/notebook.ipynb](notebook/notebook.ipynb) in VS Code or Jupyter and select the project's Python environment. It explains the real preprocessing and training pipeline, compares the ML candidates, and demonstrates how the API loads and uses a fitted model. The walkthrough uses the small synthetic demo dataset and does not overwrite the project's model artifacts.

## Setup (Windows PowerShell)

Run commands from this directory (`Week-5\Project`):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

By default, `GEMINI_MODE=mock` uses clearly labeled local demo rules and does not contact an external service. For actual Gemini classification, set `GEMINI_MODE=live`, provide `GEMINI_API_KEY`, and choose a supported `GEMINI_MODEL` in `.env`. Never commit `.env` or send sensitive customer information to an external model without authorization.

Prepare and train the included sample:

```powershell
python scripts/preprocess_data.py
python scripts/train_models.py
```

Start the API in one terminal:

```powershell
python -m uvicorn app.main:app --reload
```

Open the API docs at <http://127.0.0.1:8000/docs>. Start Streamlit in another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run frontend/app.py
```

Optional: set `TICKET_API_URL` to use an API hosted somewhere other than `http://127.0.0.1:8000`.

## Use a licensed dataset

No third-party ticket dataset or raw customer data is bundled. The included `data/raw/demo_tickets.csv` is synthetic smoke-test data only; it is too small to support meaningful quality claims or production use.

Before using an external dataset, review its license, usage terms, and privacy conditions. Do not redistribute raw data unless its license permits it. Set `TICKET_DATASET_URL`, `DATASET_SOURCE_NAME`, and `DATASET_LICENSE` in `.env` (or pass the equivalent CLI options) and run:

```powershell
python scripts/download_dataset.py
```

For a Hugging Face datasets-server JSON URL, the downloader saves the full API response as JSON and a flattened CSV under `data/external/`. It records the source URL, name, license, and row count in a neighboring `.metadata.json` file. Those downloaded files are ignored by Git.

The customer-support link above extracts 100 rows to `data/external/customer_support_tickets.json`. The Streamlit app includes a compact preview of these rows.

```powershell
python scripts/preprocess_data.py --input data/external/tickets_download.csv
```

Input CSVs must contain a `ticket_text` (or `text`/`description`) column and a `category` (or `label`/`ticket_type`) column; `subject` is optional and is combined with the ticket text. Labels must match the categories above, ignoring case. If the source uses a different taxonomy, create and review a deliberate label mapping before training; the application does not guess mappings. Preprocessing rejects unsupported labels, and training rejects missing categories or classes with too few examples.

## API

`GET /health` reports whether a trained model is loaded. `POST /classify` accepts:

```json
{
  "subject": "Unable to access account",
  "description": "I cannot sign in after resetting my password."
}
```

The response contains the selected ML, Random Forest, and Gemini predictions with confidence values, plus the hybrid category, decision reason, agreement status, and human-review flag. The final hybrid decision continues to combine the selected ML model with Gemini; Random Forest is shown as an independent comparison. Run the training steps above to generate both model artifacts.

## Evaluation and tests

`python scripts/train_models.py` compares the three ML algorithms and writes validation accuracy, macro/weighted F1, precision, recall, per-class metrics, and a confusion matrix. To compare ML candidates, Gemini, and hybrid results on a held-out split, run:

```powershell
python evaluation/evaluate.py
```

By default this uses the clearly labeled offline mock for the Gemini side. Add `--live-gemini` only when authorized to send evaluation tickets to Google; this may incur API charges. The script writes per-model confusion matrices and `evaluation/outputs/hybrid_metrics.json`. The held-out split is for development only; the bundled synthetic data is not a meaningful benchmark. Evaluate on a representative, properly licensed dataset before making performance claims.

Run the focused tests:

```powershell
python -m pytest -q
```

## Project layout

```text
app/
  core/           Shared categories
  prompts/        Gemini prompt variants
  services/       ML, Gemini, and hybrid classification
  main.py         FastAPI application
frontend/         Streamlit application
data/
  raw/            Synthetic demo tickets
  external/       Downloaded datasets (ignored by Git)
scripts/          Download, preprocess, and train
evaluation/       Generated metrics and confusion matrix
models/           Generated model artifact (ignored by Git)
tests/            Focused unit tests
```

## Security and licensing

API credentials are read from environment variables, ticket text is not logged or persisted by the API, and model output is schema-validated before use. Gemini requests send ticket text to Google's service when live mode is enabled; follow your organization's data handling policies. See the repository-level [LICENSE](../../LICENSE). Dataset licensing is separate.

