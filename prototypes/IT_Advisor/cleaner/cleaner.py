"""
cleaner.py
-----------
Removes HTML artifacts, normalizes whitespace,
and prepares text for chunking or analysis.
"""
import re

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500):
    """Split text into manageable chunks for embeddings or model context."""
    words = text.split()
    chunks, current = [], []
    for word in words:
        current.append(word)
        if len(current) >= chunk_size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks
