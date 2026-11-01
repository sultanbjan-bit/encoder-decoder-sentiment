"""Use the same review cleanup for training and the app."""
import html
import re


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())
