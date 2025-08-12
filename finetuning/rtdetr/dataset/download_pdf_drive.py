import csv
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from drive_download import download_file_from_google_drive


def is_valid_url(url):
    """
    Returns True if the URL has a valid scheme and netloc.
    """
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)


def extract_drive_links(url):
    """
    Extracts Google Drive links from the given URL without crawling additional pages.
    Downloads the files as well.
    """
    drive_links = []
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url}, status code: {response.status_code}")
            return drive_links

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = tag.get("href")
            full_url = urljoin(url, href)
            if is_valid_url(full_url) and "drive.google.com" in full_url:
                drive_links.append((url, full_url))

                # Download the file
                try:
                    download_file_from_google_drive(full_url)
                except Exception as e:
                    print(f"Error downloading file from {full_url}: {e}")

    except Exception as e:
        print(f"Error while fetching {url}: {e}")

    return drive_links


def main():
    urls = [
        # "https://www.elumbreras.com.pe/admision-unmsm-2024-ii",
        # "https://www.elumbreras.com.pe/admision-unmsm-2024-i",
        # "https://www.elumbreras.com.pe/admision-unmsm-2023-ii",
        # "https://www.elumbreras.com.pe/admision-unmsm-2023-i",
        # "https://www.elumbreras.com.pe/admision-unmsm-2022-ii",
        # "https://www.elumbreras.com.pe/admision-unmsm-2022-i",
        # "https://www.elumbreras.com.pe/admision-uni-2024-2",
        # "https://www.elumbreras.com.pe/admision-uni-2024-1",
        # "https://www.elumbreras.com.pe/admision-uni-2023-2",
        # "https://www.elumbreras.com.pe/admision-uni-2023-1",
        # "https://www.elumbreras.com.pe/admision-uni-2022-2",
        # "https://www.elumbreras.com.pe/admision-uni-2022-i",
        # "https://sacooliveros.edu.pe/index.php/academias/solucionarios/callao"
        # "https://www.elumbreras.com.pe/admision-unt-2024-i",
        # "https://www.elumbreras.com.pe/admision-unt-2023-ii",
        # "https://www.elumbreras.com.pe/admision-unt-2023-i",
        # "https://www.elumbreras.com.pe/admision-unt-2022-ii",
        # "https://www.elumbreras.com.pe/admision-unt-2022-i",
        # "https://www.elumbreras.com.pe/admision-unsa-i-fase-2024",
        # "https://www.elumbreras.com.pe/admision-unsa-ii-fase-2023",
        # "https://www.elumbreras.com.pe/admision-unsa-i-fase-2023",
        # "https://www.elumbreras.com.pe/admision-unsa-ii-fase-2022",
        # "https://www.elumbreras.com.pe/admision-unsa-i-fase-2022",
        # "https://www.elumbreras.com.pe/admision-uncp-2024-i",
        # "https://www.elumbreras.com.pe/admision-uncp-2023-ii",
        # "https://www.elumbreras.com.pe/admision-uncp-2023-i",
        # "https://www.elumbreras.com.pe/admision-uncp-2022-ii",
        # "https://www.elumbreras.com.pe/admision-uncp-2022-i",
        # "https://www.elumbreras.com.pe/admision-uncp-2021-ii",
        # "https://www.elumbreras.com.pe/solucionario-uncp-2021-i-area-ii",
        "https://www.elumbreras.com.pe/admision-unc-2024-ii",
        "https://www.elumbreras.com.pe/admision-unc-2024-i",
        "https://www.elumbreras.com.pe/admision-unc-2023-ii",
        "https://www.elumbreras.com.pe/admision-unc-2023-i",
        "https://www.elumbreras.com.pe/admision-unc-2022-ii",
        "https://www.elumbreras.com.pe/admision-unc-2022-i",
        "https://www.elumbreras.com.pe/admision-unc-2021-ii",
    ]
    for url in urls:
        start_url = url
        # start_url = "https://elumbreras.com.pe/admision-unmsm-2025-i"

        # Extract Google Drive links only from the start URL
        drive_links = extract_drive_links(start_url)

        # Save the results into a CSV file
        output_file = "google_drive_links_2.csv"
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Source Page", "Google Drive Link"])
            for source, link in drive_links:
                writer.writerow([source, link])

        print(f"\nSaved {len(drive_links)} Google Drive link(s) to '{output_file}'.")


if __name__ == "__main__":
    main()
