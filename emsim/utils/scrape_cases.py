"""
Scrape emsimcases.com by category, dedupe, and save per-category URL lists.
Stage 1: collect case URLs. Stage 2 (later): fetch .docx URLs and download.
"""
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://emsimcases.com"
CATEGORIES = [
    ("Cardiology", "cardiology"),
    ("Communication", "communication"),
    ("Endocrine", "endocrine"),
    ("GI", "gi"),
    ("Neurology", "neurology"),
    ("OB-GYN", "ob-gyn"),
    ("Pediatrics", "pediatrics"),
    ("Respiratory", "respiratory"),
    ("Resuscitation", "resuscitation"),
    ("Toxicology", "toxicology"),
    ("Trauma", "trauma"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (scraper for personal archival)"}
OUT = Path(r"D:\wmed\emsim")
OUT.mkdir(parents=True, exist_ok=True)


def fetch(url):
    for i in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text
        except Exception as e:
            print(f"  retry {i+1} for {url}: {e}")
            time.sleep(2)
    return None


def extract_case_links(html):
    """Find post-permalink links on a category archive page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    # WordPress archive posts: <h1 class="entry-title"><a href="...">Title</a></h1>
    for h in soup.select("h1.entry-title a, h2.entry-title a"):
        href = h.get("href", "")
        title = h.get_text(strip=True)
        # Filter to blog post URLs of form /YYYY/MM/DD/slug/
        if re.search(r"/\d{4}/\d{2}/\d{2}/", href):
            links.append((title, href))
    return links


def has_older_posts(html):
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select("a"):
        txt = a.get_text(strip=True).lower()
        if "older" in txt:
            return True
    return False


def scrape_category(slug):
    all_cases = []
    page = 1
    while True:
        if page == 1:
            url = f"{BASE}/category/{slug}/"
        else:
            url = f"{BASE}/category/{slug}/page/{page}/"
        print(f"  page {page}: {url}")
        html = fetch(url)
        if html is None:
            break
        links = extract_case_links(html)
        if not links:
            break
        all_cases.extend(links)
        if not has_older_posts(html):
            break
        page += 1
        time.sleep(0.5)
    return all_cases


def main():
    seen_urls = set()
    result = {}  # category -> [(title, url)]
    all_mapping = []  # full list with categories
    for cat_name, slug in CATEGORIES:
        print(f"\n=== {cat_name} ({slug}) ===")
        cases = scrape_category(slug)
        uniq = []
        for title, url in cases:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            uniq.append((title, url))
        result[cat_name] = uniq
        print(f"  {len(cases)} total, {len(uniq)} new after dedup")
        for t, u in uniq:
            all_mapping.append({"category": cat_name, "title": t, "url": u})

    (OUT / "case_index.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "case_flat.json").write_text(
        json.dumps(all_mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nTotal unique cases: {len(seen_urls)}")
    print(f"Written to {OUT / 'case_index.json'}")


if __name__ == "__main__":
    main()
