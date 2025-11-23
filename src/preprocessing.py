# src/preprocessing.py
import re
import spacy
from typing import List

# load a small model by default, user can change to 'en_core_web_sm' installed in their env
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # lazy fallback; spacy model may not be installed in testing environments
    nlp = None

def clean_text(text: str) -> str:
    """
    Basic cleaning: lowercasing, remove excessive punctuation, normalize whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # keep alphanumerics and basic punctuation
    text = re.sub(r"[^a-z0-9\s&\-/.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize(text: str) -> str:
    """
    Lemmatize using spaCy if available, otherwise return cleaned text.
    """
    text = clean_text(text)
    if nlp:
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc if not token.is_space]
        return " ".join(lemmas)
    return text

def extract_signals(text: str) -> List[str]:
    """
    Merchant-cue heuristics: split tokens, keep tokens >2 chars, and certain keywords.
    """
    cleaned = clean_text(text)
    tokens = [t for t in re.split(r"[\s/.\-]", cleaned) if len(t) > 2]
    return tokens
