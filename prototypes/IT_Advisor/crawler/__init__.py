"""
crawler package — responsible for legal, robots.txt-compliant data extraction.

Modules:
    - legal_crawler.py : Handles polite web crawling and parsing via httpx + selectolax
    - cleaner.py       : Cleans raw HTML/text and performs semantic chunking
"""

__all__ = ["legal_crawler", "cleaner"]
