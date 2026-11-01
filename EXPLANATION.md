# Understanding the project

**What does the app do?** It reads an English movie review and predicts positive
or negative sentiment. A user can select either model or compare both.

**Why two models?** The assignment asks us to compare two Transformer
architectures on the same task. Using identical test reviews makes the results
easier to compare.

**How is an encoder different from a decoder?** DistilBERT can attend to words
on both sides of a token. DistilGPT2 uses causal attention: a token attends only
to itself and earlier tokens. Its last non-padding token summarizes the prefix
for its classification head.

**Is the decoder generating text here?** No. We use its pretrained causal
backbone with a two-class prediction head. Fine-tuning adjusts the backbone and
the head for sentiment classification.

**What is tokenization?** The tokenizer converts text into token IDs that the
model can process. Each model uses its own tokenizer. Padding makes examples
in a batch the same length; an attention mask identifies real tokens. GPT-2 has
no default padding token, so we use its end-of-text token for padding and set
the corresponding model configuration.

**Why three data splits?** Training updates weights. Validation chooses the
best checkpoint. The held-out test split estimates performance after that
choice. Looking repeatedly at test results to tune the model weakens that estimate.

**What do the metrics mean?** Accuracy is the fraction of correct predictions.
Precision measures how often a predicted class is correct. Recall measures how
many actual examples of a class were found. F1 combines precision and recall.
Macro averaging gives equal weight to the two classes. A confusion matrix
shows correct and incorrect predictions separately for each class.

**Which model is better?** Answer using your generated test table. Do not choose
a winner before training. Explain tradeoffs and the limits of a small single run.

**Why Hugging Face and Streamlit?** Hugging Face stores the trained weights,
configuration and tokenizer. Streamlit provides the interface and downloads
those artifacts for inference. GitHub holds the app source used for deployment.

## Presentation outline

1. Describe the sentiment task and the assignment requirements.
2. Explain the dataset and three separate splits.
3. Explain bidirectional versus causal attention.
4. Show the training settings and actual result table.
5. Demonstrate each model and comparison mode in the live app.
6. Explain one mistake, the truncation limit, and one possible improvement.

Add your own observations after running the experiment. Keep the actual logs,
model repository links, and public app link available for questions.
