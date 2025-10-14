import re

def clean_text(text: str) -> str:
    """Cleans and normalizes user input text."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9,.!?;:'\"()\\-\\s]", "", text)
    return text.lower()
