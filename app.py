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
st.title("Movie Review Sentiment")
st.write("Compare an encoder and a decoder trained to recognize positive and negative movie reviews.")


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


choice = st.selectbox("Choose a model", list(CONFIG["models"]) + ["Compare both"])
review = st.text_area("Your movie review", placeholder="The acting was convincing and the story kept me interested.",
                      height=150, max_chars=10000)
st.caption("Only the first 128 tokens are used. Scores are model estimates, not guarantees. There is no neutral class.")
if st.button("Analyze review", type="primary"):
    text = clean_text(review)
    if not text:
        st.warning("Please enter a review first.")
    else:
        names = list(CONFIG["models"]) if choice == "Compare both" else [choice]
        rows = []
        for name in names:
            try:
                with st.spinner(f"Running {name}..."):
                    result = predict(CONFIG["models"][name]["repo"], text)
                rows.append({"Model": name, **result})
            except (OSError, ValueError, RuntimeError):
                st.error(f"{name} could not load or run. Check that its trained model repository is public and complete. Check the deployment logs for details.")
            finally:
                if choice == "Compare both":
                    load_model.clear()
                    gc.collect()
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if len(rows) == 2:
            st.info("The models agree." if rows[0]["Prediction"] == rows[1]["Prediction"] else "The models disagree on this review.")

with st.expander("Measured test results"):
    results_path = ROOT / "results" / "comparison.csv"
    if results_path.exists():
        st.dataframe(pd.read_csv(results_path), hide_index=True)
        st.caption("Held-out IMDb subset. These results do not measure performance on every kind of English text.")
    else:
        st.write("Training results have not been added yet. Run the Colab notebook and copy its results folder here.")

with st.expander("About this project"):
    st.write("DistilBERT uses bidirectional attention. DistilGPT2 uses causal attention. Both have a two-class classification head and are fine-tuned on the same IMDb training subset.")
    for name, details in CONFIG["models"].items():
        st.markdown(f"[{name} on Hugging Face](https://huggingface.co/{details['repo']})")
