# Runbook

These steps assume the current directory is `Week-5\Project`.

## First run

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies and prepare local configuration:

   ```powershell
   python -m pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

3. Keep `GEMINI_MODE=mock` for an offline demo, or set it to `live` and configure `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`. Live mode requires a valid key and an internet connection.

4. Preprocess the sample data and train the selected ML model plus the separate Random Forest comparison model:

   ```powershell
   python scripts/preprocess_data.py
   python scripts/train_models.py
   ```

5. Optionally compare ML, Gemini, and hybrid metrics on a held-out split:

   ```powershell
   python evaluation/evaluate.py
   ```

   The default Gemini path is an offline demo. Use `--live-gemini` only when authorized to send evaluation tickets to Google; the calls may incur charges.

6. Launch FastAPI:

   ```powershell
   python -m uvicorn app.main:app --reload
   ```

7. In a second terminal, activate the same environment and launch Streamlit:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m streamlit run frontend/app.py
   ```

## Replace the demo dataset

Only use a dataset after checking its source, license, privacy requirements, and suitability for the ten supported categories. Configure its direct CSV URL, source name, and license in `.env`, download it, then normalize it:

```powershell
python scripts/download_dataset.py
python scripts/preprocess_data.py --input data/external/tickets_download.csv
python scripts/train_models.py
```

If labels differ from the supported categories, map them deliberately before preprocessing. Do not redistribute source data unless its terms allow it. The downloader records source attribution alongside the local file; downloaded data and model artifacts are excluded from Git.

## Verify or troubleshoot

```powershell
python -m pytest -q
```

Check `GET http://127.0.0.1:8000/health` for model readiness. If the model is missing, preprocess and train first. If live Gemini calls fail, verify the key, model name, network access, and configured API quota. The `mock` response identifies itself as local demo rules and is not a Gemini result.

