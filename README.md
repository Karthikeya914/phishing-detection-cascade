# Phishing Detection Cascade

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/)

An applied machine learning system combining **TF-IDF + Logistic Regression**, **LSTM**, and **fine-tuned DistilBERT** in an asymmetric, confidence-based cascade with automated quarantine and REST API inference.

---

## 📌 Architecture & Design Strategy

In email security, **False Negatives (letting a phishing email slip into an inbox) carry severe security consequences compared to False Positives**. Furthermore, running deep transformer inference on every incoming email is computationally expensive and introduces unnecessary latency.

This project investigates and implements **Strategy C (Security-First Confidence Cascade)**:

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
        [ Action: QUARANTINE ]                  │                 │
                                       Confidence ≥ Threshold   Confidence < Threshold
                                                │                 │
                                                ▼                 ▼
                                       [ Action: DELIVER ]   [ Escalate to DistilBERT ]
                                                                  │
                                                        ┌─────────┴─────────┐
                                                        │                   │
                                               Predicted Phishing   Predicted Legitimate
                                                        │                   │
                                                        ▼                   ▼
                                              [ Action: QUARANTINE ]  [ Action: DELIVER ]
```

### Key Mechanisms:
1. **Compute Efficiency:** The majority of standard emails (clear phishing or highly confident legitimate messages) are classified immediately by lightweight Logistic Regression.
2. **Targeted Escalation:** Only emails where Logistic Regression predicts *legitimate* with confidence below the safety threshold (e.g., `< 0.95` or `< 0.99`) are escalated to **DistilBERT** for deep context analysis.
3. **Quarantine Logging:** When an email is flagged as phishing (either by LR or DistilBERT), it is routed to a JSON-based quarantine log (`quarantine_<timestamp>.json`) containing the email content, model used, confidence score, and timestamp.
4. **Prototyped REST API:** Wrapped in **FastAPI** with Pydantic request validation inside the notebook for easy client querying.

---

## 📊 Recorded Experimental Results

### Individual Model Benchmark (Held-out Test Set)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | 98.40% | 98.02% | 98.55% | 0.9828 |
| **LSTM (from scratch)** | 97.64% | 97.68% | 97.23% | 0.9746 |
| **DistilBERT (fine-tuned)** | **99.55%** | **99.56%** | **99.46%** | **0.9951** |

### Cascade Evaluation Performance (Evaluation Split: 20,300 samples)

The cascade strategies were developed and compared on the evaluation split:

- **Strategy C (Security-First Cascade, Threshold = 0.99):**
  - **DistilBERT Escalation Rate:** 27.34% of evaluation emails
  - **Accuracy:** 98.85%
  - **Precision:** 97.85%
  - **Recall:** 99.72% *(Minimizing false negatives to just 26 out of 9,432 phishing emails)*
  - **F1 Score:** 0.9878
- **Strategy D (Selective Second Opinion, Threshold = 0.99):**
  - **DistilBERT Escalation Rate:** 54.24% | **Accuracy:** 99.48% | **Recall:** 99.31%
- **Strategy E (Asymmetric FN-Optimized Cascade):**
  - **DistilBERT Escalation Rate:** 37.62% | **Accuracy:** 99.47% | **Recall:** 99.35%

*(Note: The holdout test evaluation for the combined cascade requires pre-computed test prediction probabilities from both models, detailed in notebook Section 7.5).*

---

## 🔬 Pipeline & Modeling Workflow

1. **Exploratory Data Analysis (EDA):** Text length distributions, vocabulary contrast analysis, sender domain profiling, and URL presence patterns across phishing vs. legitimate samples.
2. **Data Cleaning & Preprocessing:** Null handling, duplicate removal, feature concatenation (`subject` + `sender` + `text` → `email`).
3. **TF-IDF + Logistic Regression:**
   - 50,000 n-gram features (unigrams + bigrams), class-weighted logistic regression.
4. **LSTM Sequence Classifier:**
   - 20,000-word vocabulary, sequence length padded/truncated to 1,024 tokens.
   - Embedding + LSTM + Dropout + Dense Sigmoid architecture.
5. **Fine-Tuned DistilBERT:**
   - Pretrained `distilbert-base-uncased` fine-tuned for sequence classification using HuggingFace `Trainer` on GPU.
6. **Cascade & Policy Optimization:** Comparing cascade strategies (Strategy C, D, E) to balance recall against transformer compute.
7. **Simulated Quarantine & FastAPI Endpoint:** Defining the `quarantine_email` archiving function and exposing `/predict` via FastAPI.

---

## 🚀 Environment & Setup

> **Note on Environment & Artifacts:**  
> This project was developed and executed in **Google Colab** with GPU acceleration. Large model weights (such as the fine-tuned DistilBERT directory and Keras LSTM `.keras` files) and intermediate tokenized datasets are stored on Google Drive and are **not tracked in this Git repository**.

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

### 3. Running the Notebook
Open the notebook in Google Colab (recommended for GPU support during DistilBERT inference) or a local Jupyter environment:
```bash
jupyter notebook Phishing_Detection_Cascade.ipynb
```

---

## ⚡ FastAPI Inference Prototype

The notebook defines and tests a FastAPI service prototype:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Phishing Email Detection API")

class EmailRequest(BaseModel):
    email: str

@app.post("/predict")
def predict(request: EmailRequest):
    return analyze_email(request.email)
```

**Sample Request & Response:**
```json
// Input:
{
  "email": "URGENT: Your bank account will be suspended today. Verify your account immediately."
}

// Output:
{
  "prediction": "PHISHING",
  "action": "QUARANTINE",
  "model_used": "Logistic Regression",
  "lr_confidence": 99.42,
  "quarantine_file": "/content/drive/MyDrive/phishing_quarantine/quarantine_20261022_120000.json"
}
```
*(When an ambiguous legitimate email triggers DistilBERT, `"model_used": "DistilBERT"` and `"bert_confidence"` are also included).*
