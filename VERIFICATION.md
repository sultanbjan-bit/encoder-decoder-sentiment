# Verification status

Verified on 2 November 2026.

- Both models were fine-tuned on Colab Tesla T4: original run 1 November, recovery run 2 November.
- Predictive metrics match between runs; training and inference timings differ.
- DistilBERT: test accuracy 83.8%, macro-F1 0.8379837984.
- DistilGPT2: test accuracy 83.9%, macro-F1 0.8389869579.
- Confusion matrices sum to 1,000 and agree with the recorded accuracies.
- The split manifest has 4,000 training, 500 validation and 1,000 test hashes; all 5,500 are distinct.
- final_models.zip was reconstructed from 18 hash-verified parts, passed ZIP integrity checks, and was extracted locally.
- Both public Hugging Face repositories contain weights, config, tokenizer and model cards.
- All 23 uploaded files match the local saved files by SHA256. See results/public_upload_verification.json for revisions and hashes.
- Live app remote reload: the deployed code loads each public repository through AutoTokenizer.from_pretrained and AutoModelForSequenceClassification.from_pretrained. Both models loaded and produced predictions.
- Positive review: encoder POSITIVE (positive score 0.983397), decoder POSITIVE (0.958226).
- Negative review in Compare both: encoder NEGATIVE (negative score 0.983558), decoder NEGATIVE (0.991924); both rows shown.
- Blank review displays: Please enter a review first.
- A long mixed review completed in Compare both; both predictions were NEGATIVE. Only the first 128 tokens are used, as documented.
- An unauthenticated HTTP request to the live app returned status 200.

App: https://sultan-encoder-decoder-sentiment.streamlit.app/

GitHub: https://github.com/sultanbjan-bit/encoder-decoder-sentiment

Encoder: https://huggingface.co/Sultan0336/distilbert-imdb-sentiment

Decoder: https://huggingface.co/Sultan0336/distilgpt2-imdb-sentiment

Evidence: results/public_upload_verification.json, results/live_app_checks.json,
evidence/live_app_comparison.jpg and evidence/Train_Encoder_Decoder_executed.ipynb.
The executed notebook preserves the original training run and results/comparison.csv
retains its timings. Hosted checkpoints came from the recovery run.

These checks establish working artifacts and live inference at verification time.
They do not establish general performance outside this IMDb subset, calibrated
probabilities, or guaranteed hosting availability.
