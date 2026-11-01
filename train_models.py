"""Training functions used by the Colab notebook. No training starts on import."""
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from text_utils import clean_text

CONFIG = json.loads(Path("project_config.json").read_text())
OUTPUT = Path("outputs")
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}


def prepare_data():
    raw = load_dataset("stanfordnlp/imdb")
    train = raw["train"].to_pandas()
    test = raw["test"].to_pandas()
    for frame in (train, test):
        frame["text"] = frame["text"].map(clean_text)
        frame["key"] = frame["text"].str.casefold()
    train = train.drop_duplicates("key")
    test = test.drop_duplicates("key")
    # Remove exact normalized overlaps before any sampling.
    test = test[~test["key"].isin(train["key"])]
    pieces = {"train": [], "validation": [], "test": []}
    for label in (0, 1):
        pool = train[train.label == label].sample(n=2250, random_state=CONFIG["seed"])
        pieces["train"].append(pool.iloc[:2000])
        pieces["validation"].append(pool.iloc[2000:])
        pieces["test"].append(test[test.label == label].sample(n=500, random_state=CONFIG["seed"]))
    splits, manifests = {}, {}
    for split, frames in pieces.items():
        frame = pd.concat(frames).sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)
        manifests[split] = [hashlib.sha256(x.encode()).hexdigest() for x in frame["key"]]
        splits[split] = Dataset.from_pandas(frame[["text", "label"]], preserve_index=False)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not set(manifests[a]) & set(manifests[b]), f"Overlap: {a}, {b}"
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "split_manifest.json").write_text(json.dumps(manifests, indent=2))
    return DatasetDict(splits)


def compute_metrics(prediction):
    predicted = np.argmax(prediction.predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        prediction.label_ids, predicted, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(prediction.label_ids, predicted),
            "macro_precision": precision, "macro_recall": recall, "macro_f1": f1}


def train_one(name, data):
    set_seed(CONFIG["seed"])
    spec = CONFIG["models"][name]
    directory = OUTPUT / spec["repo"].split("/")[1]
    tokenizer = AutoTokenizer.from_pretrained(spec["base"])
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["base"], num_labels=2, id2label=ID2LABEL,
        label2id={value: key for key, value in ID2LABEL.items()})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    encoded = data.map(lambda batch: tokenizer(batch["text"], truncation=True,
                       max_length=CONFIG["max_length"]), batched=True, remove_columns=["text"])
    args = TrainingArguments(
        output_dir=str(directory / "checkpoints"), num_train_epochs=CONFIG["epochs"],
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        per_device_train_batch_size=8, per_device_eval_batch_size=16,
        gradient_accumulation_steps=2, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=1, logging_steps=50,
        report_to="none", seed=CONFIG["seed"], data_seed=CONFIG["seed"],
        fp16=torch.cuda.is_available(), dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=encoded["train"],
                      eval_dataset=encoded["validation"], processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    started = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - started
    # The test split is evaluated only after validation selects the checkpoint.
    predictions = trainer.predict(encoded["test"], metric_key_prefix="test")
    metrics = {key: float(value) for key, value in predictions.metrics.items()}
    metrics["training_seconds"] = training_seconds
    trainer.save_model(str(directory / "model"))
    tokenizer.save_pretrained(directory / "model")
    predicted = predictions.predictions.argmax(axis=-1)
    matrix = confusion_matrix(predictions.label_ids, predicted, labels=[0, 1]).tolist()
    (directory / "model" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (directory / "model" / "confusion_matrix.json").write_text(json.dumps({"labels": list(ID2LABEL.values()), "rows": "true", "columns": "predicted", "matrix": matrix}, indent=2))
    (directory / "model" / "experiment_config.json").write_text(json.dumps(CONFIG, indent=2))
    pd.DataFrame({"true_label": predictions.label_ids, "predicted_label": predicted}).to_csv(directory / "test_predictions.csv", index=False)
    card = f'''---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: {spec["base"]}
datasets:
- stanfordnlp/imdb
---
# {name}: IMDb sentiment

Fine-tuned for Sultan0336's encoder/decoder comparison assignment.
Base model: `{spec["base"]}`. Two-class sequence classification: 0 NEGATIVE, 1 POSITIVE.
DistilBERT is bidirectional; DistilGPT2 uses causal attention with a classification head.
This checkpoint predicts sentiment; it is not a conversational generator.

## Data and training
IMDb: 4,000 training reviews, 500 validation reviews, and 1,000 held-out test reviews.
Each split is class-balanced. Validation comes from the original training split;
test comes from the original test split. Exact normalized duplicates are removed.
HTML line breaks are removed and whitespace is normalized. Maximum length: 128 tokens.
Seed 42, 2 epochs, AdamW, learning rate 0.00002, weight decay 0.01,
warmup ratio 0.1, batch size 8, accumulation 2. Best validation macro-F1 selects the checkpoint.

## Measured held-out results
```json
{json.dumps(metrics, indent=2)}
```

## Limitations
Small training sample; one seed; movie-review domain; no neutral label.
Long reviews are truncated. Sarcasm and mixed sentiment can be misclassified.
Softmax scores are not calibrated probabilities. This comparison cannot isolate architecture
from pretraining, tokenizer, and parameter-count differences.
Dataset terms: https://huggingface.co/datasets/stanfordnlp/imdb

## Load
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="{spec["repo"]}")
print(classifier("An enjoyable movie.", truncation=True, max_length=128))
```
'''
    (directory / "model" / "README.md").write_text(card, encoding="utf-8")
    del trainer, model, tokenizer, encoded, predictions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"model": name, **metrics}


def upload_and_verify():
    api = HfApi()
    assert api.whoami()["name"].casefold() == CONFIG["hf_username"].casefold(), "Log in as Sultan0336."
    verified = {}
    for name, spec in CONFIG["models"].items():
        folder = OUTPUT / spec["repo"].split("/")[1] / "model"
        assert (folder / "metrics.json").exists(), "Train both models first."
        api.create_repo(repo_id=spec["repo"], private=False, exist_ok=True)
        commit = api.upload_folder(repo_id=spec["repo"], folder_path=str(folder))
        tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=commit.oid, force_download=True)
        model = AutoModelForSequenceClassification.from_pretrained(spec["repo"], revision=commit.oid, force_download=True)
        model.eval()
        with torch.inference_mode():
            logits = model(**tokenizer("An enjoyable movie.", return_tensors="pt")).logits
        assert logits.shape == (1, 2) and torch.isfinite(logits).all()
        verified[name] = {"repo": spec["repo"], "commit": commit.oid, "remote_reload": "passed"}
        del tokenizer, model
        gc.collect()
    (OUTPUT / "upload_verification.json").write_text(json.dumps(verified, indent=2))
    return verified
