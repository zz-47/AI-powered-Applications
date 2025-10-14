# crawler/scraper.py
# Replace your existing scraper with this focused summarizer.
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
import re

HEADERS = {"User-Agent": "IT-Advisor-Scraper/1.0 (+https://example.local/)"}

def fetch_html(url, timeout=8):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception:
        return ""

def extract_text(html, max_chars=15000):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "header", "footer", "nav", "aside", "form", "noscript"]):
        t.decompose()
    text = "\n".join(s.strip() for s in soup.stripped_strings)
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]

def sentence_split(text):
    # naive sentence split
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def score_sentences_by_query(sentences, query_tokens):
    scores = []
    qset = set(query_tokens)
    for s in sentences:
        stokens = set(re.findall(r'\w+', s.lower()))
        # score = overlap + length bonus (avoid very short sentences)
        overlap = len(qset & stokens)
        score = overlap + (min(len(s), 200) / 200.0) * 0.1
        scores.append((score, s))
    scores.sort(reverse=True, key=lambda x: x[0])
    return [s for sc, s in scores]

def summarize_docs_for_query(urls, query, max_sentences=6):
    """
    Scrape given URLs and produce a short, query-focused summary (string).
    """
    if not urls:
        return ""

    qtokens = re.findall(r'\w+', query.lower())
    collected = []
    for u in urls:
        parsed = urlparse(u)
        if parsed.scheme not in ("http", "https"):
            continue
        html = fetch_html(u)
        if not html:
            continue
        text = extract_text(html)
        sents = sentence_split(text)
        scored = score_sentences_by_query(sents, qtokens)
        top = scored[:max_sentences]
        snippet = " ".join(top).strip()
        if snippet:
            collected.append(f"URL:{u}\n{snippet}")
        time.sleep(0.4)  # be polite

    # combine snippets, limit total length
    combined = "\n\n".join(collected)
    return combined[:4000]
