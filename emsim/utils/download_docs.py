"""
Stage 2: For each case in case_index.json, visit the page, find the .docx link,
and download it into emsim/docs/<category>/.
"""
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (scraper for personal archival)"}
ROOT = Path(r"D:\wmed\emsim")
DOCS_ROOT = ROOT / "docs"
INDEX = ROOT / "case_index.json"


def sanitize(name: str) -> str:
    # keep title-based names; strip characters illegal on Windows
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "", name)
    name = name.strip().strip(".")
    return name[:150] if len(name) > 150 else name


def fetch(url):
    for i in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception as e:
            print(f"  retry {i+1} for {url}: {e}")
            time.sleep(2)
    return None


def find_docx_urls(html: str):
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        low = href.lower().split("?")[0]
        if low.endswith(".docx") or low.endswith(".doc"):
            if href not in urls:
                urls.append(href)
    return urls


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    results = []
    missing = []
    for cat, cases in index.items():
        cat_dir = DOCS_ROOT / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        for title, url in cases:
            print(f"[{cat}] {title}")
            page = fetch(url)
            if page is None:
                print("  FAILED to load page")
                missing.append({"category": cat, "title": title, "url": url, "reason": "page load failed"})
                continue
            docx_urls = find_docx_urls(page.text)
            if not docx_urls:
                print("  no .docx found")
                missing.append({"category": cat, "title": title, "url": url, "reason": "no docx link"})
                continue
            for i, doc_url in enumerate(docx_urls):
                # Filename: use case title + suffix if multiple
                orig_name = unquote(urlparse(doc_url).path.rsplit("/", 1)[-1])
                ext = ".docx" if orig_name.lower().endswith(".docx") else ".doc"
                base = sanitize(title)
                if len(docx_urls) > 1:
                    base = f"{base} ({i+1})"
                out_path = cat_dir / f"{base}{ext}"
                if out_path.exists():
                    print(f"  already downloaded: {out_path.name}")
                    results.append({"category": cat, "title": title, "file": str(out_path), "docx_url": doc_url})
                    continue
                print(f"  downloading: {doc_url}")
                resp = fetch(doc_url)
                if resp is None:
                    print("  FAILED to download")
                    missing.append({"category": cat, "title": title, "url": url, "docx_url": doc_url, "reason": "download failed"})
                    continue
                out_path.write_bytes(resp.content)
                print(f"  saved -> {out_path.name} ({len(resp.content)} bytes)")
                results.append({"category": cat, "title": title, "file": str(out_path), "docx_url": doc_url})
                time.sleep(0.3)
    (ROOT / "download_manifest.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "download_missing.json").write_text(json.dumps(missing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDownloaded: {len(results)}, Missing: {len(missing)}")


if __name__ == "__main__":
    main()
