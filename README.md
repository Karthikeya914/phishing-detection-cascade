# Phishing Detection Cascade

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade email security system combining **TF-IDF + Logistic Regression**, **LSTM**, and **fine-tuned DistilBERT** in an asymmetric, confidence-based cascade with automated quarantine and REST API inference.

---

## 📌 Architecture & Design Strategy

In email security, **False Negatives (letting a phishing email slip through) carry far greater consequences than False Positives**. Furthermore, deploying heavy transformer models for every inbound email is computationally expensive and slow.

This project implements **Strategy C (Security-First Confidence Cascade)**:

```
                            [ Inbound Email ]
                                    │
                                    ▼
                      [ Logistic Regression (TF-IDF) ]
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
        Predicted Phishing                      Predicted Legitimate
               │                                         │
               ▼                                ┌────────┴────────┐
      [ Final: Phishing ]                       │                 │
                                       Confidence ≥ 95%    Confidence < 95%
                                                │                 │
                                                ▼                 ▼
                                       [ Final: Legitimate ]  [ Escalate to DistilBERT ]
                                                                  │
                                                        ┌─────────┴─────────┐
                                                        │                   │
                                                  High Conf.           Low Conf. / Ambiguous
                                                        │                   │
                                                        ▼                   ▼
                                                [ Final Decision ]   [ Quarantine Log (.json) ]
```

### Key Highlights:
1. **Low Latency & Cost Efficiency:** Over 85–90% of emails are resolved instantly by Logistic Regression at microsecond scale.
2. **Deep Semantic Understanding:** High-risk, ambiguous emails that LR flags as "legitimate" with low confidence are escalated to a fine-tuned **DistilBERT** transformer.
3. **Automated Quarantine Mechanism:** Boundary/uncertain predictions are safely archived to a JSON audit trail rather than making hard false-negative assumptions.
4. **Deployable REST API:** Wrapped in **FastAPI** with Pydantic schema validation for seamless integration into email gateways.

---

## 📊 Pipeline & Modeling Workflow

1. **Exploratory Data Analysis (EDA):** Text length distributions, vocabulary contrast analysis, sender domain profiling, and URL presence patterns.
2. **Data Cleaning & Preprocessing:** Null handling, duplicate removal, feature concatenation (`subject` + `sender` + `text` → `email`).
3. **Model 1: TF-IDF + Logistic Regression**
   - 50,000 n-gram features (unigrams + bigrams), balanced class weighting.
4. **Model 2: LSTM Sequence Classifier**
   - 20,000-word vocabulary, sequence length capped at 1024 tokens.
   - Embedding + LSTM + Dropout + Dense Sigmoid architecture.
5. **Model 3: Fine-Tuned DistilBERT**
   - Pretrained `distilbert-base-uncased` fine-tuned for sequence classification using HuggingFace `Trainer`.
6. **Cascade & Policy Optimization:** Comparing cascade strategies (Strategy C, D, E) to maximize recall while controlling transformer compute.
7. **Production API:** FastAPI service exposing `/predict` endpoint returning predictions, active model used, confidence levels, and quarantine locations.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Karthikeya914/phishing-detection-cascade.git
cd phishing-detection-cascade
```

### 2. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Notebook Walkthrough
Launch Jupyter or Google Colab and open:
```bash
jupyter notebook Phishing_Detection_Cascade.ipynb
```

---

## ⚡ API Quickstart

The pipeline can be served directly using FastAPI:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Phishing Email Detection API")

class EmailRequest(BaseModel):
    email: str

@app.post("/predict")
def predict(request: EmailRequest):
    # Runs the Strategy C cascade (LR -> DistilBERT fallback)
    result = analyze_email(request.email)
    return result
```

**Sample API Response:**
```json
{
  "prediction": "PHISHING",
  "action": "BLOCK",
  "model_used": "Logistic Regression",
  "lr_confidence": 99.42
}
```
*(If escalated to DistilBERT or quarantined, `"model_used": "DistilBERT"` and `"quarantine_file"` paths are included in the payload).*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
