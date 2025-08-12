#!/usr/bin/env python3
import os
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def download_pdf(pdf_url, output_folder):
    local_filename = os.path.join(output_folder, pdf_url.split("/")[-1])
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {local_filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {pdf_url}: {e}")


def main(url):
    output_folder = "pdf_downloads"
    os.makedirs(output_folder, exist_ok=True)

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {e}")
        sys.exit(1)

    soup = BeautifulSoup(response.content, "html.parser")
    pdf_links = []

    # Find all anchor tags with href attribute
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Check if link ends with .pdf (case-insensitive)
        if href.lower().endswith(".pdf"):
            full_link = urljoin(url, href)
            pdf_links.append(full_link)

    print(f"Found {len(pdf_links)} PDF(s) on the page.")

    for pdf_link in pdf_links:
        download_pdf(pdf_link, output_folder)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <URL>")
        sys.exit(1)

    # Example usage:
    # python download_pdf_link.py https://www.trilce.edu.pe/academia/solucionarios-pucp
    input_url = sys.argv[1]
    main(input_url)
