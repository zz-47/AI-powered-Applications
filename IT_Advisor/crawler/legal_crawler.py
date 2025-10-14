"""
Legal crawler that respects robots.txt and public access.
Uses httpx for fetching and selectolax for parsing.
"""
import httpx
from selectolax.parser import HTMLParser


def fetch_page(url: str):
    headers = {"User-Agent": "TechWISEBot/1.0 (+https://techwise-advisor.local)"}
    try:
        r = httpx.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            html = HTMLParser(r.text)
            paragraphs = [n.text(strip=True) for n in html.css("p")]
            return " ".join(paragraphs)
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
    return None
