
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from lime.lime_text import LimeTextExplainer


# -------------------- CONFIG --------------------

st.set_page_config(
    page_title="Social Media Stress Analyzer",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DEVICE = torch.device("cpu")

torch.set_num_threads(2)


# -------------------- MODEL LOADING --------------------

@st.cache_resource
def load_logistic_regression():
    return joblib.load(
        MODEL_DIR / "tfidf_logistic_regression.joblib"
    )


@st.cache_resource
def load_random_forest():
    return joblib.load(
        MODEL_DIR / "tfidf_random_forest.joblib"
    )


@st.cache_resource
def load_transformer(folder_name):
    """Load a transformer model from the public Hugging Face repository."""
    hf_repo_id = "mommineedi/women-stress-nlp-models"

    tokenizer = AutoTokenizer.from_pretrained(
        hf_repo_id,
        subfolder=folder_name,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_repo_id,
        subfolder=folder_name,
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# -------------------- PREDICTION FUNCTIONS --------------------

def predict_classical(model, text):
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)

    negative_index = classes.index(0)
    positive_index = classes.index(1)
    predicted_index = int(np.argmax(probabilities))

    prediction = int(classes[predicted_index])
    negative_score = float(probabilities[negative_index])
    positive_score = float(probabilities[positive_index])

    return prediction, negative_score, positive_score


def predict_transformer(folder_name, text, max_length):
    tokenizer, model = load_transformer(folder_name)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[0]

    prediction = int(torch.argmax(probabilities).item())
    negative_score = float(probabilities[0].item())
    positive_score = float(probabilities[1].item())

    return prediction, negative_score, positive_score


def analyze(text, selected_model):
    if selected_model == "TF-IDF + Logistic Regression":
        model = load_logistic_regression()
        return predict_classical(model, text)

    elif selected_model == "TF-IDF + Random Forest":
        model = load_random_forest()
        return predict_classical(model, text)

    elif selected_model == "DistilBERT":
        return predict_transformer(
            "distilbert_stress", text, 96
        )

    elif selected_model == "BERT":
        return predict_transformer(
            "bert_stress", text, 64
        )

    elif selected_model == "Hybrid: LR + DistilBERT":
        lr_model = load_logistic_regression()

        _, _, lr_positive = predict_classical(
            lr_model, text
        )

        _, _, distilbert_positive = predict_transformer(
            "distilbert_stress", text, 96
        )

        hybrid_positive = (
            0.5 * lr_positive
            + 0.5 * distilbert_positive
        )
        hybrid_negative = 1.0 - hybrid_positive

        prediction = int(hybrid_positive >= 0.5)

        return prediction, float(hybrid_negative), float(hybrid_positive)

    else:
        raise ValueError("Unknown model selection.")


# -------------------- EXPLAINABLE AI: LIME --------------------

@st.cache_resource
def load_lime_explainer():
    return LimeTextExplainer(
        class_names=[
            "Stress-negative",
            "Stress-positive",
        ],
        random_state=42,
    )


def predict_probabilities_for_lime(texts, selected_model):
    """
    Return two class scores for each perturbed text:
    [P(class 0), P(class 1)].

    LIME uses these scores to estimate local feature importance.
    """
    probabilities = []

    for sample in texts:
        # Handle an empty perturbed sample safely.
        if not sample.strip():
            probabilities.append([0.5, 0.5])
            continue

        _, negative_score, positive_score = analyze(
            sample,
            selected_model,
        )

        negative_score = float(
            np.clip(negative_score, 0.0, 1.0)
        )
        positive_score = float(
            np.clip(positive_score, 0.0, 1.0)
        )

        probabilities.append([
            negative_score,
            positive_score,
        ])

    return np.asarray(probabilities, dtype=float)


def explain_prediction(text, selected_model):
    explainer = load_lime_explainer()

    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda samples: (
            predict_probabilities_for_lime(
                samples,
                selected_model,
            )
        ),
        labels=(0, 1),
        num_features=8,
        num_samples=100,
    )

    return explanation


# -------------------- USER INTERFACE --------------------

st.title("🧠 Social Media Stress Text Analyzer")

st.caption(
    "Capstone research prototype | Machine learning and transformer NLP"
)

st.warning(
    "Research prototype only. Outputs classify stress-related language; "
    "they do not diagnose a person's mental health or measure actual stress."
)


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("Model settings")

    selected_model = st.selectbox(
        "Choose a trained model",
        [
            "DistilBERT",
            "TF-IDF + Logistic Regression",
            "TF-IDF + Random Forest",
            "BERT",
            "Hybrid: LR + DistilBERT",
        ],
    )

    st.divider()

    st.subheader("Research pipeline")

    st.markdown(
        "1. Social media text\n"
        "2. Trained model inference\n"
        "3. Stress-related language classification\n"
        "4. Model score and LIME explanation"
    )

    st.caption(
        "The current dataset does not verify author gender. "
        "Predictions are not validated women-specific results."
    )


# -------------------- INPUT AND PREDICTION LAYOUT --------------------

left, right = st.columns(
    [1.1, 0.9],
    gap="large",
)

with left:
    st.subheader("1 · Enter text")

    text_input = st.text_area(
        "Social-media post or comment",
        height=200,
        placeholder=(
            "Enter or paste a de-identified text sample..."
        ),
    )

    analyze_button = st.button(
        "Analyze text",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "Avoid entering names, usernames, contact details, "
        "or other personally identifying information."
    )


with right:
    st.subheader("2 · Prediction")

    if analyze_button:
        if not text_input.strip():
            st.warning("Please enter some text first.")

        else:
            try:
                with st.spinner(
                    "Analyzing text with the selected model..."
                ):
                    prediction, negative_score, positive_score = analyze(
                        text_input.strip(),
                        selected_model,
                    )

                st.markdown(
                    f"**Selected model:** {selected_model}"
                )

                if prediction == 1:
                    predicted_label = "Stress-positive language"
                    st.error(f"Final predicted class: {predicted_label}")
                else:
                    predicted_label = "Stress-negative language"
                    st.success(f"Final predicted class: {predicted_label}")

                score_col1, score_col2 = st.columns(2)

                with score_col1:
                    st.metric(
                        "Stress-negative class score (Class 0)",
                        f"{negative_score:.1%}",
                    )

                with score_col2:
                    st.metric(
                        "Stress-positive class score (Class 1)",
                        f"{positive_score:.1%}",
                    )

                st.caption(
                    "These are the model's estimated class scores. "
                    "They describe the model's classification output, "
                    "not calibrated probabilities of a person's actual stress."
                )

                # -------------------- LIME EXPLANATION --------------------

                with st.expander(
                    "🔎 Explain this prediction (LIME)"
                ):
                    st.write(
                        "LIME estimates which words influenced this "
                        "individual model prediction by testing "
                        "perturbed versions of the input text."
                    )

                    try:
                        with st.spinner(
                            "Generating explanation... "
                            "This may take a little while on CPU."
                        ):
                            explanation = explain_prediction(
                                text_input.strip(),
                                selected_model,
                            )

                        explanation_label = (
                            "Stress-positive language"
                            if prediction == 1
                            else "Stress-negative language"
                        )

                        st.markdown(
                            f"**Explanation for:** {explanation_label}"
                        )

                        feature_weights = explanation.as_list(
                            label=prediction
                        )

                        explanation_df = pd.DataFrame(
                            feature_weights,
                            columns=[
                                "Word or phrase",
                                "LIME contribution",
                            ],
                        )

                        st.dataframe(
                            explanation_df,
                            hide_index=True,
                            use_container_width=True,
                        )

                        st.caption(
                            "Positive contributions support the selected "
                            "class; negative contributions oppose it. "
                            "LIME explanations are local approximations, "
                            "not proof of causation or clinical explanations."
                        )

                    except Exception as lime_error:
                        st.error(
                            f"Could not generate the LIME explanation: "
                            f"{lime_error}"
                        )

                        st.info(
                            "The prediction was successful. Try another "
                            "model or a shorter text sample, and check "
                            "the terminal for the full error if needed."
                        )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

                st.caption(
                    "Check that the saved model files exist in the "
                    "models folder and that the required packages "
                    "are installed in the active virtual environment."
                )

    else:
        st.info(
            "Enter text and click Analyze text to get a prediction."
        )


# -------------------- MODEL INFORMATION --------------------


# -------------------- PROTOTYPE RESULTS --------------------

st.divider()
st.subheader("3 · Prototype Results")

st.caption(
    "Final evaluation on 715 test samples"
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Best test accuracy",
        "76.78%",
        help="Achieved by DistilBERT."
    )

with col2:
    st.metric(
        "Best test F1-score",
        "77.75%",
        help="Achieved by DistilBERT."
    )

results_df = pd.DataFrame({
    "Model": [
        "TF-IDF + Logistic Regression",
        "TF-IDF + Random Forest",
        "DistilBERT",
        "BERT",
        "Hybrid: LR + DistilBERT",
    ],
    "Accuracy": [
        "72.17%",
        "69.23%",
        "76.78%",
        "68.67%",
        "75.10%",
    ],
    "F1-score": [
        "74.05%",
        "72.29%",
        "77.75%",
        "72.48%",
        "76.27%",
    ],
    "ROC-AUC": [
        "81.88%",
        "77.42%",
        "83.63%",
        "74.44%",
        "84.42%",
    ],
})

st.dataframe(
    results_df,
    hide_index=True,
    use_container_width=True,
)

st.markdown("**Key finding:**")

st.write(
    "DistilBERT achieved the highest test accuracy and F1-score "
    "among the evaluated models. The LR + DistilBERT hybrid "
    "achieved the highest ROC-AUC. These results indicate that "
    "DistilBERT performed best on the selected classification "
    "metrics, while the hybrid showed stronger ranking performance."
)

st.caption(
    "These metrics measure classification of stress-related language, "
    "not actual psychological stress or clinical outcomes. "
    "The dataset does not establish validated women-specific performance."
)
