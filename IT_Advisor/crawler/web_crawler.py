import requests
from bs4 import BeautifulSoup

def scrape_web_summary(query: str) -> str:
    """
    Scrapes short contextual info from Google search results for additional reasoning.
    Returns summarized text snippets.
    """
    try:
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}+IT+development"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(search_url, headers=headers, timeout=5)

        soup = BeautifulSoup(resp.text, "html.parser")
        snippets = [tag.get_text() for tag in soup.select("div.BNeawe.s3v9rd.AP7Wnd")]
        return " ".join(snippets[:3]) if snippets else ""
    except Exception:
        return ""
