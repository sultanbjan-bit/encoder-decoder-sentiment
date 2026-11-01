# Encoder and Decoder Sentiment Assignment

Prepared for GitHub user **sultanbjan-bit** and Hugging Face user **Sultan0336**.

## Current status

Both models were fine-tuned and evaluated on a Colab Tesla T4 on 1 November 2026.
DistilBERT achieved 83.8% test accuracy and 0.837984 macro-F1; DistilGPT2 achieved
83.9% test accuracy and 0.838987 macro-F1. The one-review accuracy difference does
not establish a clear winner. See results/ and the executed notebook for evidence.
Hugging Face uploads and public deployment are in progress.

## Start here

1. Open Google Colab: https://colab.research.google.com/ . Select **File > Upload notebook** and upload `Train_Encoder_Decoder.ipynb`.
2. Choose **Runtime > Change runtime type > T4 GPU** (or another available GPU).
3. Run the cells in order. The notebook contains the Python source and config, so you do not need to upload extra files.
4. After training, inspect the comparison table. The first model-loading warning about a newly initialized classification head is expected: training learns that head.
5. Run the login cell and enter a Hugging Face write token in its private widget. Create the token in your Hugging Face account settings. Never paste it into a code cell or chat.
6. Run the upload cell. It publishes both trained models to your account, then downloads each uploaded revision and tests it.
7. Download the results ZIP. Keep a copy of the notebook with outputs for evidence. Download the separate model backup ZIP if desired.

If Colab offers no GPU, wait for availability or use another GPU runtime. The notebook stops before expensive CPU training. Runtime storage is temporary: download artifacts before disconnecting. Interrupted training can require rerunning the affected model. Do not report unfinished runs as final results.

## Deploy the app

1. Sign in to GitHub as `sultanbjan-bit`. Create a repository named `encoder-decoder-sentiment`.
2. Upload `app.py`, `text_utils.py`, `project_config.json`, and `requirements.txt` at the repository root. Also upload this README, the notebook and training source for reproducibility.
3. Extract the Colab results ZIP. Upload its `results` folder to the repository root, including `comparison.csv`, `report.md`, and verification records. Do not upload model checkpoints or tokens to GitHub.
4. Confirm both Hugging Face repositories are public and contain `config.json`, tokenizer files, model weights, and README files.
5. Sign in at https://share.streamlit.io/ with GitHub and create an app from this repository, branch `main`, entrypoint `app.py`. Select Python 3.11 under advanced settings.
6. Deploy. The first inference needs to download the selected model. The app runs models sequentially to reduce memory use.
7. Test the live URL with the encoder, decoder, and Compare both options. Also test a blank input, a long review, a negative review, and a mixed review. Confirm both outputs appear in comparison mode. Record failures honestly; disagreement is not itself an app failure.
8. Open the URL in a private browser window to confirm public access. Submit that verified URL with both model links and the GitHub repository link.

Expected model repository names (created only by the notebook upload step):
- https://huggingface.co/Sultan0336/distilbert-imdb-sentiment
- https://huggingface.co/Sultan0336/distilgpt2-imdb-sentiment

For local inference, install `requirements.txt` in a Python 3.11 environment and run `streamlit run app.py`. The models must already be uploaded.

## Experiment design

We use the Stanford IMDb dataset, with **4,000 training**, **500 validation**,
and **1,000 test** reviews. Each split has equal positive and negative counts.
The notebook downloads the original dataset but trains on this smaller subset.
It removes exact normalized duplicates before sampling and stores split hashes.
The seed is 42. Validation comes only from the original training split. Test
reviews come only from the original test split and do not select the checkpoint.

Both models use the same reviews, maximum length of 128 tokens, learning rate
of 0.00002, two epochs, and effective training batch size of 16. Different
tokenizers can retain different amounts of text within 128 tokens. The best
validation macro-F1 selects each checkpoint. We report held-out accuracy,
macro precision, macro recall, macro-F1 and confusion matrices. Training time
depends on the hardware and should not be treated as a controlled speed benchmark.

DistilBERT is an encoder with bidirectional attention. DistilGPT2 is a smaller
GPT-2-family decoder with causal attention. We fine-tune both backbones plus
their new classification heads. The decoder is used for classification, not
free-text generation; the assignment permits predictions or generated outputs.

This is a small, single-seed classroom comparison. Differences cannot be
attributed only to architecture: pretraining, tokenizer, parameter count, and
optimization also differ. Softmax scores are not calibrated confidence. The
system has no neutral class, truncates long reviews, and may struggle with
sarcasm, mixed opinions, and other domains.

## Files and evidence

- `Train_Encoder_Decoder.ipynb`: standalone Colab workflow.
- `train_models.py`: readable training, evaluation, upload and reload functions.
- `app.py`: one Streamlit app for both models.
- `project_config.json`: account names, model IDs and shared settings.
- `text_utils.py`: shared cleanup used during training and inference.
- `EXPLANATION.md`: short presentation and viva guide.
- `results/`: generated by a successful training run, not fabricated in advance.

## Sources

- Assignment: supplied two-page PDF, “Larger language model Encoder only & Decoder only”.
- IMDb dataset and terms: https://huggingface.co/datasets/stanfordnlp/imdb
- Original dataset: https://ai.stanford.edu/~amaas/data/sentiment/
- DistilBERT model card: https://huggingface.co/distilbert/distilbert-base-uncased
- DistilGPT2 model card: https://huggingface.co/distilbert/distilgpt2
- GPT-2 sequence classification: https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/gpt2
- Deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

Read the notebook and adapt the explanation to your understanding. Follow your
course's rules on acknowledging assistance. Do not claim that code checks prove
training, model accuracy, or successful live deployment.
