# 🧠 Transformer-Based NLP Framework for Women-Focused Social Media Analytics

### An Explainable AI Approach to Stress-Related Language Classification

<p align="center">
  <a href="https://women-stress-nlp-prototype.streamlit.app/">🚀 Live Application</a> •
  <a href="https://github.com/navyasree1385/women-stress-nlp-prototype">💻 GitHub Repository</a> •
  <a href="https://huggingface.co/mommineedi/women-stress-nlp-models">🤗 Hugging Face Models</a>
</p>

---

## 📌 Project Overview

The **Transformer-Based NLP Framework for Women-Focused Social Media Analytics** is a machine learning and Natural Language Processing (NLP) research prototype designed to classify stress-related language in social media text.

The system analyzes user-provided text and predicts whether it contains stress-positive or stress-negative language using traditional machine learning algorithms, transformer-based architectures, and a hybrid classification approach.

The application also integrates **LIME (Local Interpretable Model-Agnostic Explanations)** to highlight words and phrases that influence individual predictions, improving the interpretability of the model outputs.

The project provides an interactive Streamlit application where users can enter text, select a trained model, view class scores, and explore explanations.

## 🎯 Objectives

- Develop an NLP-based framework for stress-related language classification.
- Compare traditional machine learning algorithms with transformer-based models.
- Implement a hybrid model combining Logistic Regression and DistilBERT.
- Evaluate model performance using standard classification metrics.
- Integrate LIME for explainable AI.
- Develop and deploy an interactive web application for real-time text analysis.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| NLP / Transformers | Hugging Face Transformers |
| Explainable AI | LIME |
| Model Persistence | Joblib |
| Web Application | Streamlit |
| Model Hosting | Hugging Face Hub |
| Source Code Management | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## 🧠 Models Implemented

Five classification configurations were developed and evaluated.

| Model | Description |
|---|---|
| TF-IDF + Logistic Regression | Traditional machine learning baseline using TF-IDF text features. |
| TF-IDF + Random Forest | Ensemble-based classification using TF-IDF features. |
| DistilBERT | Lightweight transformer-based language classification model. |
| BERT | Bidirectional transformer-based language classification model. |
| Hybrid: Logistic Regression + DistilBERT | Combines probability outputs from Logistic Regression and DistilBERT using equal weighting. |

### Hybrid Model

The hybrid approach combines the positive-class scores from Logistic Regression and DistilBERT.

**Fusion formula:**

`Hybrid Score = (0.5 × LR Score) + (0.5 × DistilBERT Score)`

The negative-class score is calculated as:

`Negative Score = 1 − Hybrid Score`

A threshold of 0.5 is used for final classification.

## 📊 Dataset Information

The project uses a Reddit-derived stress-related text classification dataset based on the Dreaddit dataset.

**Dataset summary:**

| Dataset Split | Number of Records |
|---|---:|
| Training | 2,247 |
| Validation | 570 |
| Testing | 715 |
| **Total** | **3,532** |

### Training Dataset Class Distribution

| Class | Records |
|---|---:|
| Class 0 – Stress-negative language | 1,092 |
| Class 1 – Stress-positive language | 1,155 |

### Classification Labels

- **Class 0:** Stress-negative language
- **Class 1:** Stress-positive language

**Dataset limitation:** The available dataset does not independently verify the gender of every author. Therefore, the project explores a women-focused research application but does not claim that the dataset or model has been validated exclusively on women's social media data.

## 📈 Model Evaluation Results

All five configurations were evaluated on the same test set containing 715 samples.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 72.17% | 71.36% | 76.96% | 74.05% | 81.88% |
| TF-IDF + Random Forest | 69.23% | 67.53% | 77.78% | 72.29% | 77.42% |
| DistilBERT | **76.78%** | **76.92%** | 78.59% | **77.75%** | 83.63% |
| BERT | 68.67% | 66.29% | **79.95%** | 72.48% | 74.44% |
| Hybrid: LR + DistilBERT | 75.10% | 75.07% | 77.51% | 76.27% | **84.42%** |

### Key Findings

- **DistilBERT** achieved the highest accuracy (76.78%) and F1-score (77.75%).
- **Hybrid LR + DistilBERT** achieved the highest ROC-AUC (84.42%).
- **BERT** achieved the highest recall (79.95%).
- DistilBERT demonstrated the strongest overall performance based on accuracy and F1-score among the evaluated configurations.

## 🔍 Explainable AI – LIME

The project integrates LIME to provide local explanations for individual predictions.

LIME analyzes perturbed versions of the input text to estimate the contribution of individual words or phrases to the selected model's prediction.

**Explanation features:**

- Identifies influential words and phrases.
- Displays positive and negative feature contributions.
- Provides explanations for individual text predictions.
- Helps users understand model behavior.

Positive LIME contributions support the class being explained, while negative contributions oppose it.

LIME explanations are local approximations and do not establish causation.

## 🌐 Web Application Features

The Streamlit application provides an interactive interface with the following capabilities:

1. **Model Selection:** Choose between five trained classification configurations.
2. **Text Input:** Enter or paste social media text.
3. **Text Classification:** Predict stress-positive or stress-negative language.
4. **Class Scores:** Display scores for both classification classes.
5. **LIME Explanation:** Explore the words influencing the prediction.
6. **Model Performance:** View the comparative evaluation results.
7. **Research Information:** Understand the research pipeline and project limitations.

### Live Application

🚀 https://women-stress-nlp-prototype.streamlit.app/

## 🏗️ Project Architecture

```text
                User Input
                    |
                    v
             Streamlit Interface
                    |
                    v
              Model Selection
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
      TF-IDF     DistilBERT     BERT
        |           |           |
        v           |           |
   Logistic /       |           |
   Random Forest    |           |
        |           |           |
        +-----------+-----------+
                    |
                    v
             Hybrid Prediction
                    |
                    v
          Class Scores and Label
                    |
                    v
              LIME Explanation
                    |
                    v
              Final Output
```

## 📂 Repository Structure

```text
women-stress-nlp-prototype/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── tfidf_logistic_regression.joblib
│   ├── tfidf_random_forest.joblib
│   ├── bert_stress/
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   │
│   ├── distilbert_stress/
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   │
│   ├── final_test_comparison.csv
│   └── final_test_results.json
│
├── train_baseline.py
├── train_random_forest.py
├── train_bert.py
├── train_distilbert.py
├── compare_models.py
├── evaluate_test.py
├── evaluate_hybrid.py
├── evaluate_hybrid_all.py
├── evaluate_hybrid_lr_bert.py
│
├── check_dataset.py
├── check_transformer_version.py
└── test_transformer.py
```

**Note:** The large BERT and DistilBERT model weight files are hosted separately on Hugging Face Hub rather than being stored in the GitHub repository.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/navyasree1385/women-stress-nlp-prototype.git
```

### 2. Navigate to the project directory

```bash
cd women-stress-nlp-prototype
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will be available locally at:

`http://localhost:8501`

## ☁️ Deployment Architecture

The project uses three platforms for source code management, model hosting, and application deployment.

| Platform | Purpose |
|---|---|
| GitHub | Source code and project files |
| Hugging Face Hub | BERT and DistilBERT model files |
| Streamlit Community Cloud | Public web application hosting |

**Deployment links:**

- GitHub: https://github.com/navyasree1385/women-stress-nlp-prototype
- Hugging Face: https://huggingface.co/mommineedi/women-stress-nlp-models
- Streamlit: https://women-stress-nlp-prototype.streamlit.app/

## ⚠️ Limitations and Disclaimer

- This is an academic research prototype, not a clinical diagnostic system.
- Predictions classify stress-related language and do not measure actual psychological stress.
- The dataset does not verify the gender of every author.
- Model predictions may be affected by context, sarcasm, ambiguity, and variations in language.
- LIME provides local feature approximations rather than causal explanations.
- The reported evaluation results represent performance on the selected test dataset and should not be interpreted as clinical validation.

## 🚀 Future Enhancements

- Develop and evaluate a gender-verified and ethically curated dataset.
- Explore multilingual and context-aware NLP models.
- Improve classification performance through further experimentation.
- Investigate advanced explainability techniques such as SHAP.
- Evaluate generalization across different social media platforms.
- Improve model efficiency and inference speed.
- Conduct broader validation using independently collected datasets.

## 👩‍💻 Project Information

**Project Title:** Transformer-Based NLP Framework for Women-Focused Social Media Analytics

**Domain:** Natural Language Processing, Machine Learning, Deep Learning, Explainable AI

**Application:** Social Media Stress-Related Language Classification

**Development Tools:** Python, PyTorch, Hugging Face Transformers, Scikit-learn, Streamlit and LIME

---

<p align="center">
  <b>Developed as an academic capstone research project.</b>
</p>
