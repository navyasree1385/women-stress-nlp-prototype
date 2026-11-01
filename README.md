# Women-Focused Social Media Stress Analyzer — Prototype

## Important
This is a Streamlit interface for a Review 2 demonstration. Its current prediction uses a keyword-based placeholder, not a trained BERT/RoBERTa or hybrid model. The score is not a probability, the explanation is not SHAP/LIME, and this is not a clinical tool.

## Run locally
1. Install Python 3.10 or newer.
2. Extract the ZIP and open a terminal in this folder.
3. Run:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

4. Open the local URL shown in the terminal.

## Before claiming research results
- Verify dataset source, license/terms, text fields, labels, and collection method.
- Document annotation guidelines and quality checks for newly labeled data.
- Prevent leakage in train/validation/test splits.
- Train baselines and the proposed method on held-out data.
- Implement and document the actual fusion strategy and explanation method.
- Never present demo outputs as validated model results.
