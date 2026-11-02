"""Run with: streamlit run app.py"""
import gc
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from text_utils import clean_text

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "project_config.json").read_text())
st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="centered")
st.title("🎬 Movie Review Sentiment")
st.write("Find out whether a movie review sounds positive or negative. Choose a model, paste your review, and select Analyze review.")


@st.cache_resource(max_entries=1)
def load_model(repo):
    # Keep just one model cached to reduce memory use on the hosted app.
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo)
    model.eval()
    return tokenizer, model


def predict(repo, text):
    tokenizer, model = load_model(repo)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=CONFIG["max_length"])
    with torch.inference_mode():
        scores = torch.softmax(model(**inputs).logits, dim=-1)[0]
    index = int(scores.argmax())
    return {"Prediction": model.config.id2label[index],
            "Positive score": float(scores[1]), "Negative score": float(scores[0])}


def model_label(name):
    if name == "Compare both":
        return "Compare both models"
    architecture, model = name.split(": ", 1)
    return f"{model} ({architecture.lower()})"


with st.form("review_form"):
    choice = st.selectbox("1. Choose a model", list(CONFIG["models"]) + ["Compare both"],
                          format_func=model_label,
                          help="Choose one model or compare their answers on the same review.")
    review = st.text_area("2. Enter your movie review", placeholder="I loved this movie. The story was engaging and the acting was excellent.",
                          height=150, max_chars=10000)
    submitted = st.form_submit_button("Analyze review", type="primary", use_container_width=True)
st.caption("Long reviews are shortened to the first 128 tokens (roughly words or word parts). Results update when you select Analyze review.")
if submitted:
    text = clean_text(review)
    if not text:
        st.warning("Please enter a review first.")
    else:
        names = list(CONFIG["models"]) if choice == "Compare both" else [choice]
        rows = []
        for name in names:
            try:
                with st.spinner(f"Analyzing with {model_label(name)}..."):
                    result = predict(CONFIG["models"][name]["repo"], text)
                rows.append({"Model": name, **result})
            except (OSError, ValueError, RuntimeError):
                st.error(f"{model_label(name)} is unavailable right now. Please try again in a moment.")
            finally:
                if choice == "Compare both":
                    load_model.clear()
                    gc.collect()
        if rows:
            st.subheader("Your result" if len(rows) == 1 else "Your results")
            columns = st.columns(len(rows))
            for column, row in zip(columns, rows):
                with column:
                    with st.container(border=True):
                        st.markdown(f"**{model_label(row['Model'])}**")
                        positive = row["Prediction"] == "POSITIVE"
                        label = "Positive" if positive else "Negative"
                        selected_score = row[f"{label} score"]
                        if positive:
                            st.success("Positive review", icon="👍")
                        else:
                            st.error("Negative review", icon="👎")
                        st.metric(f"{label} score", f"{selected_score:.1%}")
                        st.progress(row["Positive score"], text=f"Positive: {row['Positive score']:.1%}")
                        st.progress(row["Negative score"], text=f"Negative: {row['Negative score']:.1%}")
            st.caption("Percentages show each model's scores for this review. They are estimates, not guaranteed accuracy. The model chooses the higher score; there is no neutral category.")
        if len(rows) == 2:
            st.info("The models agree." if rows[0]["Prediction"] == rows[1]["Prediction"] else "The models disagree on this review.")

with st.expander("How accurate are the models?"):
    results_path = ROOT / "results" / "comparison.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        display_results = pd.DataFrame({
            "Model": results["model"].map(model_label),
            "Test accuracy": results["test_accuracy"].map(lambda value: f"{value:.1%}"),
            "Macro-F1": results["test_macro_f1"].map(lambda value: f"{value:.4f}"),
        })
        st.dataframe(display_results, hide_index=True, use_container_width=True)
        st.caption("Measured on 1,000 held-out IMDb reviews. These overall test results are different from the score for your individual review.")
    else:
        st.write("Training results have not been added yet. Run the Colab notebook and copy its results folder here.")

with st.expander("About this project"):
    st.write("DistilBERT uses bidirectional attention. DistilGPT2 uses causal attention. Both have a two-class classification head and are fine-tuned on the same IMDb training subset.")
    for name, details in CONFIG["models"].items():
        st.markdown(f"[{name} on Hugging Face](https://huggingface.co/{details['repo']})")

