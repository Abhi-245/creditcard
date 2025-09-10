import asyncio
import os
import time
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from PyPDF2 import PdfReader
import io
import re

# -------------------------------
# Helpers
# -------------------------------

def fetch_with_retries(url, retries=3, timeout=15):
    """Fetch a webpage (or detect PDF) with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Edg/139.0.0.0"},
                allow_redirects=True,
            )
            content_type = resp.headers.get("Content-Type", "").lower()

            if resp.url.lower().endswith(".pdf") or "application/pdf" in content_type:
                return "pdf", resp.content

            if resp.status_code == 200:
                return "html", resp.text

            print(f"[WARN] {url} returned {resp.status_code} (attempt {attempt})")
        except Exception as e:
            print(f"[ERROR] Fetch failed for {url} (attempt {attempt}): {e}")

        time.sleep(2 * attempt)  # backoff

    return None, None


def extract_pdf_bytes(pdf_bytes):
    """Extract text from PDF bytes."""
    try:
        file_like = io.BytesIO(pdf_bytes)
        reader = PdfReader(file_like)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
        return ""

import hashlib
# def safe_filename_from_url(url: str, folder: str = "") -> str:
#     safe = re.sub(r"[^\w\-_.]", "_", url)
#     if folder:
#         os.makedirs(folder, exist_ok=True)
#         return os.path.join(folder, f"{safe}.txt")
#     return f"{safe}.txt"
# import hashlib

def safe_filename_from_url(url: str, folder: str = "") -> str:
    """
    Create a short, safe filename from a URL using MD5 hash.
    """
    hashed = hashlib.md5(url.encode("utf-8")).hexdigest()
    safe = f"{hashed}.txt"
    if folder:
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, safe)
    return safe


# -------------------------------
# Main crawler
# -------------------------------

async def crawl_and_scrape_all(start_url):
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        print("\n=== Crawling Main Page ===")
        result = await crawler.arun(start_url, config=run_config)

        if not result.success:
            print(f"[ERROR] Failed to crawl main page: {result.error_message}")
            return

        html_source = result.html or ""
        soup = BeautifulSoup(html_source, "html.parser")

        # Extract all links
        links_set = set()
        for a in soup.find_all("a", href=True):
            href = a.get("href").strip()
            if href and not href.startswith("#") and not href.lower().startswith("javascript:"):
                links_set.add(urljoin(result.url, href))

        # Extract full text
        full_text = soup.get_text(separator="\n", strip=True)

        # Save main page
        main_filename = safe_filename_from_url(result.url, folder="scraped_main")
        with open(main_filename, "w", encoding="utf-8") as f:
            f.write("URL: " + result.url + "\n\n")
            f.write("Links found:\n")
            f.write("\n".join(sorted(links_set)) + "\n\n")
            f.write("Full page text:\n")
            f.write(full_text or "(no text extracted)\n")
        print(f"Saved main page -> {main_filename}")

        # Scrape each link
        combined_texts = [f"==== Main Page: {result.url} ====\n{full_text}\n"]
        for link in sorted(links_set):
            print(f"\n[Scraping] {link}")
            kind, data = fetch_with_retries(link)
            if not kind:
                print(f"[FAILED] Could not scrape {link}")
                continue

            if kind == "pdf":
                text = extract_pdf_bytes(data)
            else:
                soup2 = BeautifulSoup(data, "html.parser")
                text = soup2.get_text(separator="\n", strip=True)

            # Save individual page
            link_file = safe_filename_from_url(link, folder="scraped_links")
            with open(link_file, "w", encoding="utf-8") as f:
                f.write("URL: " + link + "\n\n")
                f.write(text or "(no text extracted)")

            combined_texts.append(f"==== Linked Page: {link} ====\n{text or '(no text extracted)'}\n")
            print(f"Saved linked page -> {link_file}")

        # Save combined
        with open("combined_scraped_output.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(combined_texts))
        print("\nSaved combined output to combined_scraped_output.txt")


# -------------------------------
# Run
# -------------------------------

if __name__ == "__main__":
    asyncio.run(crawl_and_scrape_all(
        "https://www.icicibank.com/personal-banking/cards/credit-card/makemytrip/makemytrip-icici-credit-card"
    ))
