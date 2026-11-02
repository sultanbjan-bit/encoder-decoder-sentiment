# Encoder-only and Decoder-only Assignment

Student: Sultan Jan

## Submission links

- Live app: https://sultan-encoder-decoder-sentiment.streamlit.app/
- Code and executed notebook: https://github.com/sultanbjan-bit/encoder-decoder-sentiment
- Encoder: https://huggingface.co/Sultan0336/distilbert-imdb-sentiment
- Decoder: https://huggingface.co/Sultan0336/distilgpt2-imdb-sentiment

## Work completed

DistilBERT and DistilGPT2 were fine-tuned for positive/negative movie-review
classification on the same balanced IMDb subset: 4,000 training, 500 validation
and 1,000 held-out test reviews. Exact normalized duplicates were removed. The
best checkpoint was selected by validation macro-F1.

| Model | Test accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|
| DistilBERT (encoder) | 83.8% | 0.838135 | 0.838000 | 0.837984 |
| DistilGPT2 (decoder) | 83.9% | 0.839110 | 0.839000 | 0.838987 |

DistilGPT2 classified one more test review correctly. This small difference in
one seed does not show that either architecture is generally better. Both models
use a classification head; DistilGPT2 is not used for free-text generation here.

The app supports either model and Compare both. Both hosted models loaded and
ran. Positive, negative, blank and long mixed review checks were completed on
2 November 2026. Detailed checks are in VERIFICATION.md.

## Files to read

- results/report.md: method, evaluation and interpretation.
- EXPLANATION.md: short guide for explaining the assignment.
- Train_Encoder_Decoder.ipynb: reproducible training and upload workflow.
- evidence/Train_Encoder_Decoder_executed.ipynb: original training outputs.

The source package excludes the large model weights, which are public on Hugging
Face and also saved separately in final_models.zip. Review the code and explanation
before presenting and follow course rules for acknowledging assistance.
