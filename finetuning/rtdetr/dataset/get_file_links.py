import csv
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def is_valid_url(url):
    """
    Returns True if the URL has a valid scheme and netloc.
    """
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)


def crawl(url, domain, visited, drive_links, max_depth=2, current_depth=0):
    """
    Recursively crawl pages starting at `url` if they belong to the `domain`.
    Extracts links that contain 'drive.google.com' and follows internal links up to `max_depth`.
    The drive_links list will contain tuples of (source_page, google_drive_link).
    """
    if current_depth > max_depth:
        return

    if url in visited:
        return

    print(f"[Depth {current_depth}] Crawling: {url}")
    visited.add(url)
    try:
        response = requests.get(url, timeout=10)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return

    if response.status_code != 200:
        return

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all anchor tags with an href attribute
    for tag in soup.find_all("a", href=True):
        href = tag.get("href")
        full_url = urljoin(url, href)
        if not is_valid_url(full_url):
            continue

        # If the link is a Google Drive link, record it along with the current page.
        if "drive.google.com" in full_url:
            drive_links.append((url, full_url))

        # If the link is part of the same domain, crawl it further.
        parsed_href = urlparse(full_url)
        if domain in parsed_href.netloc:
            crawl(full_url, domain, visited, drive_links, max_depth, current_depth + 1)

    # Be polite with a short delay between requests.
    time.sleep(1)


def main():
    start_url = "https://elumbreras.com.pe/admision-unmsm-2025-i"
    domain = "elumbreras.com.pe"
    visited = set()
    drive_links = []  # List to store tuples of (source_page, drive_link)

    crawl(start_url, domain, visited, drive_links, max_depth=2)

    # Save the results into a CSV file.
    output_file = "google_drive_links.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Source Page", "Google Drive Link"])
        for source, link in drive_links:
            writer.writerow([source, link])

    print(
        f"\nSaved {len(drive_links)} Google Drive link(s) with associated source pages to '{output_file}'."
    )


if __name__ == "__main__":
    main()
