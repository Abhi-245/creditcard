"""
optimized_crawler.py
-------------------
Crawls a main page with crawl4ai and follow-up links with requests + BeautifulSoup.
Shows progress bars and saves all content safely.
"""

import asyncio
import os
import re
from urllib.parse import urldefrag, urljoin
import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from tqdm import tqdm

def safe_filename_from_url(url: str, suffix: str = "", folder: str = "") -> str:
    safe = re.sub(r'[^\w\-_.]', '_', url)
    if suffix:
        safe = f"{safe}_{suffix}"
    if folder:
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{safe}.txt")
    return f"{safe}.txt"

def filter_valid_links(links, base_url):
    cleaned = []
    for l in links:
        if not l.startswith("http"):
            continue
        if l.lower().endswith(".pdf"):
            continue
        if "#" in l:
            continue
        if l.rstrip("/") == base_url.rstrip("/"):
            continue
        if "javascript:" in l.lower() or "mailto:" in l.lower():
            continue
        cleaned.append(l)
    return list(sorted(set(cleaned)))

async def crawl_content_sections(start_urls, max_concurrent=20):
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)

    visited = set()
    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = [normalize_url(u) for u in start_urls]

    combined_texts = []  # for combinedscrapping.txt

    async with AsyncWebCrawler(config=browser_config) as crawler:
        print("\n=== Crawling Main Pages ===")
        urls_to_crawl = [u for u in current_urls if u not in visited]
        if not urls_to_crawl:
            print("No URLs to crawl.")
            return

        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config)
        if not results:
            print("No results returned from arun_many.")
            return

        # --- Main Pages ---
        for result in tqdm(results, desc="Main pages"):
            url = normalize_url(result.url)
            visited.add(url)
            if not result.success:
                print(f"[ERROR] {result.url}: {result.error_message}")
                continue

            html_source = result.html or ""
            links_set = set()
            section_texts = []

            # Parse main page content
            try:
                import lxml.html as lxml_html
                tree = lxml_html.fromstring(html_source)
                nodes = []
                nodes.extend(tree.xpath('/html/body/div[1]/section[4]'))
                nodes.extend(tree.xpath('/html/body/div[1]/section[5]'))
                nodes.extend(tree.xpath('/html/body/div[1]/section[6]/div/section'))
                if not nodes:
                    nodes = tree.xpath('//div[contains(@class,"content")]') or [tree]

                for node in nodes:
                    text = node.text_content().strip()
                    if text:
                        section_texts.append(text)
                    for a in node.xpath('.//a[@href]'):
                        href = a.get('href')
                        if href and not href.startswith('#') and not href.lower().startswith('javascript:'):
                            links_set.add(urljoin(result.url, href))

            except Exception:
                soup = BeautifulSoup(html_source, "html.parser")
                nodes = []
                sel1 = soup.select('body > div:nth-of-type(1) > section:nth-of-type(4)')
                sel2 = soup.select('body > div:nth-of-type(1) > section:nth-of-type(5)')
                sel3 = soup.select('body > div:nth-of-type(1) > section:nth-of-type(6) > div > section')
                nodes = sel1 + sel2 + sel3 or soup.find_all("div", class_="content") or [soup]

                for node in nodes:
                    text = node.get_text(separator="\n", strip=True)
                    if text:
                        section_texts.append(text)
                    for a in node.find_all("a", href=True):
                        href = a.get("href").strip()
                        if href and not href.startswith('#') and not href.lower().startswith('javascript:'):
                            links_set.add(urljoin(result.url, href))

            # Save main page
            combined_text = "\n\n".join(section_texts).strip()
            combined_texts.append(f"==== Main Page: {result.url} ====\n{combined_text}\n")
            txt_filename = safe_filename_from_url(result.url, suffix="sections")
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write("URL: " + result.url + "\n\nLinks:\n")
                f.write("\n".join(sorted(links_set)) if links_set else "(none)")
                f.write("\n\n--- Extracted Section Text ---\n\n")
                f.write(combined_text or "(no content extracted)\n")

            # --- Follow-up links ---
            valid_links = filter_valid_links(links_set, result.url)
            print(f"\nValid links to follow ({len(valid_links)}).")

            for link in tqdm(valid_links, desc=f"Scraping links for {url[:30]}"):
                try:
                    safe_url = safe_filename_from_url(link, folder="linkbylinkscrap")
                    if link.lower().endswith(".pdf"):
                        with open(safe_url, "w", encoding="utf-8") as f:
                            f.write(f"PDF link: {link}\n")
                        combined_texts.append(f"==== PDF Link: {link} ====\n(Saved as PDF reference)\n")
                        continue

                    resp = requests.get(link, timeout=10)
                    resp.raise_for_status()
                    soup2 = BeautifulSoup(resp.text, "html.parser")
                    text2 = soup2.get_text(separator="\n", strip=True)

                    with open(safe_url, "w", encoding="utf-8") as f:
                        f.write("URL: " + link + "\n\n")
                        f.write(text2)

                    combined_texts.append(f"==== Linked Page: {link} ====\n{text2}\n")
                except Exception as e:
                    print(f"[ERROR] Failed to fetch {link}: {e}")

    # --- Save combined file ---
    with open("combinedscrapping.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined_texts))
    print("\nSaved combined output to combinedscrapping.txt")

# --- Run ---
if __name__ == "__main__":
    asyncio.run(crawl_content_sections(
        ["https://www.sbicard.com/en/personal/credit-cards/rewards/cashback-sbi-card.page"],
        max_concurrent=20
    ))
