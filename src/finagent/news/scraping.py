import json
import re
from bs4 import BeautifulSoup

EXTRACTION_VERSION = 2


def _split_blocks(blocks):
    first, extra, origins, extra_origins = [], [], [], []
    for index, block in enumerate(blocks):
        fragments = re.split(r"(?:<br\s*/?>\s*){2,}|</p>\s*", str(block), flags=re.I)
        texts = [BeautifulSoup(fragment, "lxml").get_text(" ", strip=True) for fragment in fragments]
        texts = [text for text in texts if text]
        first.append(texts[0] if texts else "")
        origins.append({"wrapper_index": index, "fragment_index": 0})
        for subindex, text in enumerate(texts[1:], 1):
            extra.append(text)
            extra_origins.append({"wrapper_index": index, "fragment_index": subindex})
    return {"paragraphs": first + extra, "block_origins": origins + extra_origins,
            "extraction_version": EXTRACTION_VERSION}


def extract_article_blocks(html: str, expected_nid: int) -> dict:
    soup = BeautifulSoup(html, "lxml")
    blocks = soup.find_all("div", class_="newsTextDataWrapInner")
    if blocks:
        return {"extraction_format": "newsTextDataWrapInner",
                **_split_blocks(blocks)}
    
    script = soup.find("script", id="__NEXT_DATA__")
    if script:
        article = json.loads(script.string).get("props", {}).get("pageProps", {}).get("data", {})
        if article.get("nid") != int(expected_nid):
            raise ValueError("Article payload identity mismatch")
        fragment = BeautifulSoup(article.get("content", ""), "html.parser")
        blocks = fragment.find_all("div", class_="gmail_default", recursive=False)
        paragraphs = [block.get_text(" ", strip=True) for block in blocks]
        paragraphs = [text for text in paragraphs if text]
        if paragraphs:
            return {"extraction_format": "public_content_gmail_default",
                    **_split_blocks([block for block in blocks if block.get_text(" ", strip=True)])}
    raise ValueError(f"No supported public article blocks for {expected_nid}")
