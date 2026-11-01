# Encoder and Decoder Sentiment Comparison

## Method
We fine-tuned DistilBERT and DistilGPT2 on the same balanced IMDb subset: 4,000 training, 500 validation and 1,000 held-out test reviews. Both backbones and classification heads were updated. The best validation macro-F1 checkpoint was selected for each model. Test data was used only after checkpoint selection.

Settings: seed 42, two epochs, learning rate 2e-5, effective batch size 16, maximum 128 tokens. Exact normalized duplicates were removed before sampling.

## Actual test results
| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Encoder: DistilBERT | 0.8380 | 0.8381 | 0.8380 | 0.8380 |
| Decoder: DistilGPT2 | 0.8390 | 0.8391 | 0.8390 | 0.8390 |

## Interpretation
DistilGPT2 had higher test macro-F1 by 0.0010 in this run.
This one run does not establish that either architecture is universally better. Tokenizers, pretraining and model size differ. Movie-review results may not transfer to other domains. Long reviews are truncated and neutral sentiment is unsupported.

## Deployment evidence
Hugging Face upload verification and live Streamlit checks must be added after they succeed. Record the public app URL and examples you actually tested.

## Sources
- https://huggingface.co/datasets/stanfordnlp/imdb
- https://huggingface.co/distilbert/distilbert-base-uncased
- https://huggingface.co/distilbert/distilgpt2