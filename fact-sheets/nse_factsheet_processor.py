"""
NSE Index Factsheet Processor

This script downloads and parses NSE (National Stock Exchange of India) index factsheets
using pdfplumber for efficient table extraction. It handles all steps from downloading
the factsheets to parsing them and saving structured data in JSON format.
"""

import requests
import os
import time
import json
import logging
import concurrent.futures
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
import pdfplumber
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)
from tqdm import tqdm
from pathlib import Path
import argparse


# Configure logging
def get_logger():
    """Create and configure logger with proper encoding for Unicode symbols"""
    log_formatter = logging.Formatter(
        "%(levelname)s:%(name)s:%(asctime)s.%(msecs)d %(filename)s:%(lineno)d:%(funcName)s() %(message)s",
        datefmt="%A,%d/%m/%Y|%H:%M:%S",
    )

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler("factsheet_processor.log", encoding="utf-8")
    file_handler.setFormatter(log_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    # Create logger
    logger = logging.getLogger("factsheet_processor")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = get_logger()


class NSEFactsheetDownloader:
    """Download NSE index factsheets from the NSE India website."""

    def __init__(self, output_folder="nifty_factsheets"):
        self.base_url = "https://niftyindices.com"
        # List of all category pages to scan
        self.category_urls = [
            "https://niftyindices.com/indices/equity/broad-based-indices",
            "https://niftyindices.com/indices/equity/sectoral-indices",
            "https://niftyindices.com/indices/equity/thematic-indices",
            "https://niftyindices.com/indices/equity/strategy-indices",
        ]
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        # Headers to mimic a browser request
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type((requests.RequestException, IOError)),
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.DEBUG),
    )
    def _make_request(self, url):
        """Make an HTTP request with retry logic"""
        response = self.session.get(url, timeout=30)
        if response.status_code == 200:
            return response
        elif response.status_code == 404:
            logger.warning(f"Resource not found (404): {url}")
            return None
        else:
            logger.warning(f"HTTP error {response.status_code} for {url}")
            response.raise_for_status()

    def _find_factsheet_links(self):
        """Find factsheet links from the NSE India website by scanning all category pages"""
        logger.info(f"Searching for factsheet links across all category pages")

        try:
            factsheet_links = []
            index_links = []

            # Loop through all category pages
            for category_url in self.category_urls:
                logger.info(f"Processing category: {category_url}")

                # Get all index links from the category page
                response = self._make_request(category_url)
                if not response:
                    logger.warning(f"Could not access category page: {category_url}")
                    continue

                soup = BeautifulSoup(response.content, "html.parser")

                # Extract the category path from URL (like "/indices/equity/thematic-indices")
                category_path = urlparse(category_url).path

                # Find all individual index links for this category
                for link in soup.find_all("a", href=True):
                    # Check if link belongs to this category and is not the category page itself
                    if category_path in link["href"] and not link["href"].endswith(
                        category_path.split("/")[-1]
                    ):
                        absolute_url = urljoin(self.base_url, link["href"])
                        if absolute_url not in index_links:
                            index_links.append(absolute_url)

                # Add a small delay to avoid overwhelming the server
                time.sleep(0.5)

            logger.info(
                f"Found {len(index_links)} individual index pages to check across all categories"
            )

            # Visit each individual index page to find factsheet links
            for index_link in index_links:
                logger.debug(f"Checking for factsheet on index page: {index_link}")
                try:
                    index_response = self._make_request(index_link)
                    if not index_response:
                        continue

                    index_soup = BeautifulSoup(
                        index_response.content, "html.parser"
                    )  # Look for factsheet links on individual index pages using multiple methods

                    # Method 1: Check for any links that have 'factsheet' in href or text
                    factsheet_found = False

                    # Debug: Log all link hrefs on this page
                    all_pdf_links = []
                    for link in index_soup.find_all("a", href=True):
                        if link["href"].lower().endswith(".pdf"):
                            all_pdf_links.append(link["href"])

                    if all_pdf_links:
                        logger.info(
                            f"Found {len(all_pdf_links)} PDF links on page: {index_link}"
                        )
                        logger.info(f"PDF links: {all_pdf_links}")

                    # First check: Look for links with 'factsheet' in href or text that end with .pdf
                    for link in index_soup.find_all("a", href=True):
                        if "factsheet" in link["href"].lower() and link[
                            "href"
                        ].lower().endswith(".pdf"):
                            absolute_url = urljoin(self.base_url, link["href"])
                            if absolute_url not in factsheet_links:
                                factsheet_links.append(absolute_url)
                                factsheet_found = True
                                logger.info(
                                    f"Found factsheet link by href: {absolute_url}"
                                )

                        # Also check link text for "factsheet" mention
                        elif "factsheet" in link.text.lower() and link[
                            "href"
                        ].lower().endswith(".pdf"):
                            absolute_url = urljoin(self.base_url, link["href"])
                            if absolute_url not in factsheet_links:
                                factsheet_links.append(absolute_url)
                                factsheet_found = True
                                logger.debug(
                                    f"Found factsheet link by text: {absolute_url}"
                                )

                    # Second check: If no factsheet links found, look for downloads section or any buttons
                    if not factsheet_found:
                        # Look for download sections or buttons
                        download_sections = index_soup.find_all(
                            ["div", "section", "ul", "li"],
                            {
                                "class": lambda c: c
                                and ("download" in c.lower() or "reports" in c.lower())
                            },
                        )

                        for section in download_sections:
                            for link in section.find_all("a", href=True):
                                if link["href"].lower().endswith(".pdf"):
                                    absolute_url = urljoin(self.base_url, link["href"])
                                    if absolute_url not in factsheet_links:
                                        factsheet_links.append(absolute_url)
                                        logger.info(
                                            f"Found PDF in download section: {absolute_url}"
                                        )

                    # Third check: If still no links, look for any buttons or anchors that might be download buttons
                    if not factsheet_found and not download_sections:
                        download_buttons = index_soup.find_all(
                            "a",
                            {
                                "class": lambda c: c
                                and any(
                                    keyword in c.lower()
                                    for keyword in ["download", "btn", "button"]
                                )
                            },
                        )

                        for button in download_buttons:
                            if button.get("href") and button["href"].lower().endswith(
                                ".pdf"
                            ):
                                absolute_url = urljoin(self.base_url, button["href"])
                                if absolute_url not in factsheet_links:
                                    factsheet_links.append(absolute_url)
                                    logger.debug(
                                        f"Found PDF through button: {absolute_url}"
                                    )

                    # Fourth check: As a last resort, just look for any PDF on the page if it's likely a factsheet
                    # based on the index name in the URL
                    if not factsheet_found:
                        index_name = os.path.basename(index_link).lower()

                        for link in index_soup.find_all("a", href=True):
                            if link["href"].lower().endswith(".pdf"):
                                pdf_name = os.path.basename(link["href"]).lower()

                                # If the PDF name contains parts of the index name, it's likely the factsheet
                                if any(
                                    part in pdf_name for part in index_name.split("-")
                                ):
                                    absolute_url = urljoin(self.base_url, link["href"])
                                    if absolute_url not in factsheet_links:
                                        factsheet_links.append(absolute_url)
                                        logger.debug(
                                            f"Found potential factsheet by name matching: {absolute_url}"
                                        )

                    # Add a small delay to avoid overwhelming the server
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Error processing index page {index_link}: {str(e)}")
                    continue  # If we still couldn't find any factsheet links, try the reports/index-factsheet page
            if not factsheet_links:
                logger.info(
                    "No factsheets found on individual pages, checking main factsheet page"
                )

                # Try the main index factsheet page
                factsheet_page_url = "https://niftyindices.com/reports/index-factsheet"
                logger.info(
                    f"Checking main factsheet repository at {factsheet_page_url}"
                )
                factsheet_page = self._make_request(factsheet_page_url)

                if factsheet_page:
                    factsheet_soup = BeautifulSoup(
                        factsheet_page.content, "html.parser"
                    )

                    # Debug: Check all PDF links on this page
                    all_pdf_links = []
                    for link in factsheet_soup.find_all("a", href=True):
                        if link["href"].lower().endswith(".pdf"):
                            all_pdf_links.append(link["href"])

                    if all_pdf_links:
                        logger.info(
                            f"Found {len(all_pdf_links)} PDF links on main factsheet page"
                        )
                        logger.info(f"Sample PDF links: {all_pdf_links[:5]}")

                    # First look for links in tables or lists
                    for link in factsheet_soup.find_all("a", href=True):
                        if link["href"].lower().endswith(".pdf"):
                            absolute_url = urljoin(self.base_url, link["href"])
                            if absolute_url not in factsheet_links:
                                factsheet_links.append(absolute_url)
                                logger.info(
                                    f"Found factsheet on main page: {absolute_url}"
                                )

                # Also try a few alternate places
                alternate_urls = [
                    "https://www.niftyindices.com/factsheet",
                    "https://niftyindices.com/reports/index-factsheet",
                    "https://niftyindices.com/factsheet",
                    "https://www.nseindia.com/products-services/indices-factsheets",
                    "https://web.archive.org/web/20230601000000*/https://www1.nseindia.com/products/content/equities/indices/fact_sheet.htm",
                ]

                for alt_url in alternate_urls:
                    try:
                        logger.info(f"Trying alternate factsheet URL: {alt_url}")
                        alt_response = self._make_request(alt_url)
                        if alt_response:
                            alt_soup = BeautifulSoup(
                                alt_response.content, "html.parser"
                            )
                            for link in alt_soup.find_all("a", href=True):
                                if link["href"].lower().endswith(".pdf"):
                                    absolute_url = urljoin(alt_url, link["href"])
                                    if absolute_url not in factsheet_links:
                                        factsheet_links.append(absolute_url)
                                        logger.info(
                                            f"Found factsheet on alternate page: {absolute_url}"
                                        )
                    except Exception as e:
                        logger.warning(
                            f"Error checking alternate URL {alt_url}: {str(e)}"
                        )

            # Filter to only include factsheets, not methodology PDFs
            factsheet_links = [
                link for link in factsheet_links if "methodology" not in link.lower()
            ]

            logger.info(f"Found {len(factsheet_links)} potential factsheet links")
            return factsheet_links

        except Exception as e:
            logger.error(f"Error finding factsheet links: {str(e)}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, IOError)),
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.DEBUG),
    )
    def download_factsheet(self, url):
        """Download a single factsheet PDF"""
        filename = os.path.basename(urlparse(url).path)
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        # Skip methodology files
        if "method" in filename.lower():
            logger.debug(f"Skipping methodology file: {filename}")
            return None

        output_path = os.path.join(self.output_folder, filename)

        # If file already exists, return "skipped" to indicate it wasn't downloaded but exists
        if os.path.exists(output_path):
            logger.debug(f"Factsheet already exists: {output_path}")
            return "skipped"

        logger.debug(f"Downloading factsheet: {url}")

        try:
            response = self._make_request(url)
            if not response:
                return None

            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.debug(f"Downloaded factsheet to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error downloading factsheet {url}: {str(e)}")
            return None

    def download_all_factsheets(self):
        """Download all available factsheets"""
        factsheet_links = self._find_factsheet_links()

        # Add direct factsheet URLs for key indices that might not be found through the regular process
        # These are common index factsheets that should be available
        direct_factsheet_urls = [
            "https://www1.nseindia.com/content/indices/ind_nifty_allshare.pdf",
            "https://www1.nseindia.com/content/indices/ind_nifty50_factsheet.pdf",
            "https://www1.nseindia.com/content/indices/ind_niftynext50_factsheet.pdf",
            "https://www1.nseindia.com/content/indices/ind_nifty100_factsheet.pdf",
            "https://www1.nseindia.com/content/indices/ind_nifty200_factsheet.pdf",
            "https://archives.nseindia.com/content/indices/ind_nifty50_factsheet.pdf",
            "https://niftyindices.com/reports/Factsheet/2023/monthly/Nifty%20Indices%20Monthly%20Report_Feb%202023.pdf",
            "https://iislliveblob.niftyindices.com/reports/Factsheet/2022/monthly/Nifty-Indices-Monthly-Report_Dec-2022.pdf",
            "https://www.nseindia.com/content/indices/NIFTY_50.pdf",
        ]

        # Add these direct URLs to our list
        for url in direct_factsheet_urls:
            if url not in factsheet_links:
                factsheet_links.append(url)
                logger.info(f"Added direct factsheet URL: {url}")

        # Summary statistics
        stats = {
            "total_links": len(factsheet_links),
            "successful_downloads": 0,
            "failed_downloads": 0,
            "skipped_downloads": 0,
            "categories_found": len(self.category_urls),
        }

        if not factsheet_links:
            logger.warning(
                "No factsheet links found. Checking central factsheet repository as last resort."
            )
            # Try the central factsheet repository
            factsheet_page_url = "https://niftyindices.com/reports/index-factsheet"
            factsheet_page = self._make_request(factsheet_page_url)

            if factsheet_page:
                factsheet_soup = BeautifulSoup(factsheet_page.content, "html.parser")
                for link in factsheet_soup.find_all("a", href=True):
                    if link["href"].lower().endswith(".pdf"):
                        absolute_url = urljoin(self.base_url, link["href"])
                        if (
                            absolute_url not in factsheet_links
                            and "methodology" not in absolute_url.lower()
                        ):
                            factsheet_links.append(absolute_url)

                logger.info(
                    f"Found {len(factsheet_links)} factsheet links from central repository"
                )
                stats["total_links"] = len(factsheet_links)

        # Now download all factsheet PDFs
        downloaded_files = []

        # Use ThreadPoolExecutor for parallel downloads with better progress reporting
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.download_factsheet, url): url
                for url in factsheet_links
            }

            # Use tqdm to display progress
            with tqdm(total=len(futures), desc="Downloading factsheets") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    url = futures[future]
                    try:
                        result = future.result()
                        if result:
                            if result == "skipped":
                                stats["skipped_downloads"] += 1
                                logger.debug(f"Skipped (already exists): {url}")
                            else:
                                stats["successful_downloads"] += 1
                                downloaded_files.append(result)
                                logger.debug(f"Successfully downloaded: {url}")
                        else:
                            stats["failed_downloads"] += 1
                            logger.warning(f"Failed to download: {url}")
                    except Exception as e:
                        stats["failed_downloads"] += 1
                        logger.error(f"Error downloading {url}: {str(e)}")
                    pbar.update(1)

        # Log summary statistics
        logger.info("=" * 50)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Categories scanned: {stats['categories_found']}")
        logger.info(f"Total factsheet links found: {stats['total_links']}")
        logger.info(f"Successfully downloaded: {stats['successful_downloads']}")
        logger.info(f"Already existing (skipped): {stats['skipped_downloads']}")
        logger.info(f"Failed downloads: {stats['failed_downloads']}")
        logger.info("=" * 50)

        return downloaded_files


class PDFFactsheetParser:
    """Parse NSE index factsheets using pdfplumber for table extraction."""

    def __init__(self, output_folder="parsed_factsheets"):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def parse_factsheet(self, pdf_path):
        """Parse a single factsheet PDF using pdfplumber"""
        logger.info(f"Parsing factsheet: {pdf_path}")

        result = {
            "file_name": os.path.basename(pdf_path),
            "metadata": {},
            "portfolio_characteristics": {},
            "sector_representation": {},
            "returns": {},
            "statistics": {},
            "fundamentals": {},
            "top_constituents": [],
            "success": False,
        }

        try:
            # Open the PDF with pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                result["metadata"] = {
                    "total_pages": len(pdf.pages),
                }

                # Extract text from all pages
                full_text = ""
                all_tables = []

                for i, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"

                    # Extract tables - pdfplumber is excellent at identifying tables
                    tables = page.extract_tables()
                    if tables:
                        logger.debug(f"Found {len(tables)} tables on page {i+1}")
                        for table in tables:
                            # Convert to pandas DataFrame for easier processing
                            # Filter out completely empty rows and columns
                            filtered_table = []
                            for row in table:
                                if any(cell and cell.strip() for cell in row if cell):
                                    filtered_table.append(
                                        [cell if cell else "" for cell in row]
                                    )

                            if filtered_table:
                                # Convert to DataFrame and clean up
                                df = pd.DataFrame(filtered_table)
                                # Assume first row is header if it doesn't contain numbers
                                if df.shape[0] > 0 and not any(
                                    re.search(r"\d+", str(cell))
                                    for cell in df.iloc[0]
                                    if cell
                                ):
                                    df.columns = df.iloc[0]
                                    df = df.iloc[1:].reset_index(drop=True)

                                # Clean up column names
                                df.columns = [
                                    str(col).strip() if col else f"Column_{i}"
                                    for i, col in enumerate(df.columns)
                                ]

                                # Add to our tables list
                                all_tables.append(df)

                # Save first page text for metadata
                if full_text:
                    result["metadata"]["first_page_text"] = full_text[
                        :500
                    ]  # Just the start for context
                # Try the specialized extraction for the specific table format in the image
                self._extract_returns_from_image_format(all_tables, result)

                # If we couldn't extract returns or statistics using specialized method, try general approach
                if not result["returns"] or not result["statistics"]:
                    # Process extracted tables for relevant information
                    self._extract_from_tables(all_tables, result)

                    # Extract data from text using regex patterns
                    self._extract_from_text(full_text, result)

                # Post-process results
                self._post_process_results(result)

            # Set success flag if we extracted meaningful data
            total_items = sum(
                len(result[k]) for k in result if isinstance(result[k], dict)
            )
            total_items += len(result["top_constituents"])
            result["success"] = total_items > 0

            return result

        except Exception as e:
            logger.error(f"Error parsing PDF with pdfplumber: {str(e)}")
            return result

    def _extract_returns_from_image_format(self, tables, result):
        """Extract returns data specifically from the format shown in the factsheet image"""
        logger.debug("Attempting to extract returns from image format")

        for table in tables:
            # Skip very small tables
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue

            # Convert table to string for easier identification
            table_str = table.to_string().lower()

            # Check if this looks like the returns table from the image
            # Look for "index returns (%)" format as in the example image
            if (
                "index" in table_str and "returns (%)" in table_str
            ) or "returns (%)" in table_str:
                logger.debug(f"Found returns table matching image format: {table}")

                # Process the returns table
                for idx, row in table.iterrows():
                    # Check if this is one of the return rows
                    row_str = " ".join(
                        [str(cell).lower() if cell else "" for cell in row]
                    )

                    # Look for price return or total return rows
                    if "price return" in row_str:
                        for col_idx, col in enumerate(table.columns):
                            col_lower = str(col).lower()
                            # Skip first column which is likely "Price Return" text
                            if col_idx == 0:
                                continue

                            # Map columns to period names based on position or header text
                            period = None
                            if "qtd" in col_lower:
                                period = "QTD"
                            elif "ytd" in col_lower:
                                period = "YTD"
                            elif "1 year" in col_lower:
                                period = "1 Year"
                            elif "5 year" in col_lower:
                                period = "5 Years"
                            elif "inception" in col_lower:
                                period = "Since Inception"
                            else:
                                # If no match in header, try to infer from position
                                if col_idx == 1:
                                    period = "QTD"
                                elif col_idx == 2:
                                    period = "YTD"
                                elif col_idx == 3:
                                    period = "1 Year"
                                elif col_idx == 4:
                                    period = "5 Years"
                                elif col_idx == 5:
                                    period = "Since Inception"

                            if period:
                                value = row.iloc[col_idx]
                                if value is not None:
                                    # Handle numeric values including negative values
                                    value_str = str(value)
                                    # Look for negative values specifically -N.NN format
                                    number_match = re.search(
                                        r"([-+]?[0-9]*\.?[0-9]+)", value_str
                                    )
                                    if number_match:
                                        result["returns"][f"{period} Price Return"] = (
                                            number_match.group(1)
                                        )

                    elif "total return" in row_str:
                        for col_idx, col in enumerate(table.columns):
                            col_lower = str(col).lower()
                            # Skip first column which is likely "Total Return" text
                            if col_idx == 0:
                                continue

                            # Map columns to period names based on position or header text
                            period = None
                            if "qtd" in col_lower:
                                period = "QTD"
                            elif "ytd" in col_lower:
                                period = "YTD"
                            elif "1 year" in col_lower:
                                period = "1 Year"
                            elif "5 year" in col_lower:
                                period = "5 Years"
                            elif "inception" in col_lower:
                                period = "Since Inception"
                            else:
                                # If no match in header, try to infer from position
                                if col_idx == 1:
                                    period = "QTD"
                                elif col_idx == 2:
                                    period = "YTD"
                                elif col_idx == 3:
                                    period = "1 Year"
                                elif col_idx == 4:
                                    period = "5 Years"
                                elif col_idx == 5:
                                    period = "Since Inception"

                            if period:
                                value = row.iloc[col_idx]
                                if value is not None:
                                    # Handle numeric values including negative values
                                    value_str = str(value)
                                    # Look for negative values specifically -N.NN format
                                    number_match = re.search(
                                        r"([-+]?[0-9]*\.?[0-9]+)", value_str
                                    )
                                    if number_match:
                                        result["returns"][f"{period} Total Return"] = (
                                            number_match.group(1)
                                        )

                logger.debug(f"Extracted returns: {result['returns']}")

            # Check if this looks like the statistics table from the image
            elif "statistics" in table_str:
                logger.debug(f"Found statistics table matching image format: {table}")

                # Process the statistics table
                for idx, row in table.iterrows():
                    # Convert row to string for matching
                    row_str = " ".join(
                        [str(cell).lower() if cell else "" for cell in row]
                    )

                    # Look for specific statistic rows
                    stat_name = None
                    if "std" in row_str or "deviation" in row_str:
                        stat_name = "Std. Deviation"
                    elif "beta" in row_str:
                        stat_name = "Beta (NIFTY 50)"
                    elif "correlation" in row_str:
                        stat_name = "Correlation (NIFTY 50)"

                    if stat_name:
                        for col_idx, col in enumerate(table.columns):
                            col_lower = str(col).lower()
                            # Skip first column which is likely the stat name
                            if col_idx == 0:
                                continue

                            # Map columns to period names based on position or header text
                            period = None
                            if "1 year" in col_lower:
                                period = "1 Year"
                            elif "5 year" in col_lower:
                                period = "5 Years"
                            elif "inception" in col_lower:
                                period = "Since Inception"
                            else:
                                # If no match in header, try to infer from position
                                if col_idx == 1:
                                    period = "1 Year"
                                elif col_idx == 2:
                                    period = "5 Years"
                                elif col_idx == 3:
                                    period = "Since Inception"

                            if period:
                                value = row.iloc[col_idx]
                                if value is not None:
                                    value_str = str(value)
                                    number_match = re.search(
                                        r"([-+]?[0-9]*\.?[0-9]+)", value_str
                                    )
                                    if number_match:
                                        result["statistics"][
                                            f"{stat_name} ({period})"
                                        ] = number_match.group(1)

                logger.debug(f"Extracted statistics: {result['statistics']}")

    def _extract_from_tables(self, tables, result):
        """Extract structured data from tables"""
        # Process each table
        for table in tables:
            # Skip very small tables
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue

            # Convert table to string for easier pattern matching
            table_str = table.to_string().lower()

            # Check different table types

            # 1. Check if this is a returns table
            if any(term in table_str for term in ["return", "period", "performance"]):
                self._process_returns_table(table, result)

            # 2. Check if this is a constituents table
            elif any(
                term in table_str
                for term in ["constituent", "weight", "holding", "stock", "security"]
            ):
                self._process_constituents_table(table, result)

            # 3. Check if this is a sector table
            elif any(
                term in table_str for term in ["sector", "industry", "allocation"]
            ):
                self._process_sector_table(table, result)

            # 4. Check if this is a portfolio characteristics table
            elif any(
                term in table_str
                for term in ["methodology", "base date", "base value", "launch date"]
            ):
                self._process_portfolio_table(table, result)

            # 5. Check if this is a statistics/fundamentals table
            elif any(
                term in table_str for term in ["statistic", "p/e", "p/b", "dividend"]
            ):
                self._process_stats_table(table, result)

    def _process_returns_table(self, table, result):
        """Extract returns data from a table"""
        logger.debug("Processing returns table")

        # Look for specific headers that indicate this is the returns table
        table_cols = [str(col).lower() for col in table.columns]

        # Check if this looks like the returns table format in the factsheet
        if any("qtd" in col for col in table_cols) or any(
            "ytd" in col for col in table_cols
        ):
            logger.debug("Found returns table with period columns")

            # Try to find the rows that contain 'price return' and 'total return'
            for idx, row in table.iterrows():
                row_as_str = [str(cell).lower().strip() if cell else "" for cell in row]

                # Check if this row has return type info
                if any("price return" in cell for cell in row_as_str):
                    # This is the Price Return row
                    for col_idx, col_name in enumerate(table.columns):
                        col_name_lower = str(col_name).lower().strip()

                        # Map column names to standardized period names
                        period_map = {
                            "qtd": "QTD",
                            "ytd": "YTD",
                            "1 year": "1 Year",
                            "5 years": "5 Years",
                            "since inception": "Since Inception",
                        }

                        # Find matching period
                        matched_period = None
                        for period_key, period_name in period_map.items():
                            if period_key in col_name_lower:
                                matched_period = period_name
                                break

                        if matched_period:
                            # Extract value, ensuring we capture negative numbers
                            value = row.iloc[col_idx]
                            if value and str(value).strip():
                                # Extract number including negative sign
                                number_match = re.search(
                                    r"([-+]?[0-9]*\.?[0-9]+)", str(value)
                                )
                                if number_match:
                                    result["returns"][
                                        f"{matched_period} Price Return"
                                    ] = number_match.group(1)

                elif any("total return" in cell for cell in row_as_str):
                    # This is the Total Return row
                    for col_idx, col_name in enumerate(table.columns):
                        col_name_lower = str(col_name).lower().strip()

                        # Map column names to standardized period names
                        period_map = {
                            "qtd": "QTD",
                            "ytd": "YTD",
                            "1 year": "1 Year",
                            "5 years": "5 Years",
                            "since inception": "Since Inception",
                        }

                        # Find matching period
                        matched_period = None
                        for period_key, period_name in period_map.items():
                            if period_key in col_name_lower:
                                matched_period = period_name
                                break

                        if matched_period:
                            # Extract value, ensuring we capture negative numbers
                            value = row.iloc[col_idx]
                            if value and str(value).strip():
                                # Extract number including negative sign
                                number_match = re.search(
                                    r"([-+]?[0-9]*\.?[0-9]+)", str(value)
                                )
                                if number_match:
                                    result["returns"][
                                        f"{matched_period} Total Return"
                                    ] = number_match.group(1)

        # Handle case where the table has different layout
        else:
            # If the table doesn't match the expected format, try alternative approach
            logger.debug("Using alternative approach for returns table")

            # Convert columns and rows to lowercase for case-insensitive matching
            table_lower = table.astype(str).apply(lambda x: x.str.lower())

            # Standard period names to look for
            periods = {
                "1 Year": ["1 year", "1 yr", "1y", "one year"],
                "5 Years": ["5 years", "5 yrs", "5y", "five years"],
                "QTD": ["qtd", "quarter to date"],
                "YTD": ["ytd", "year to date"],
                "Since Inception": ["since inception", "inception"],
                "Price Return": ["price return", "price"],
                "Total Return": ["total return", "total"],
            }

            # First try column-based extraction
            for col in table_lower.columns:
                col_str = str(col).lower()
                for period_key, period_patterns in periods.items():
                    if any(pattern in col_str for pattern in period_patterns):
                        # Found a column matching this period
                        # Take the first non-empty value that looks like a number
                        for value in table[col].dropna().astype(str):
                            number_match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", value)
                            if number_match:
                                result["returns"][period_key] = number_match.group(1)
                                break

            # Then try row-based extraction
            for row_idx, row in table_lower.iterrows():
                for col_idx, cell in enumerate(row):
                    cell_str = str(cell).lower() if cell else ""
                    for period_key, period_patterns in periods.items():
                        if any(pattern in cell_str for pattern in period_patterns):
                            # Found a cell matching this period
                            # Check the next few cells for a value
                            for i in range(1, min(4, len(row) - col_idx)):
                                value_cell = row.iloc[col_idx + i]
                                if value_cell and isinstance(
                                    value_cell, (str, int, float)
                                ):
                                    value_str = str(value_cell)
                                    number_match = re.search(
                                        r"([-+]?[0-9]*\.?[0-9]+)", value_str
                                    )
                                    if number_match:
                                        result["returns"][period_key] = (
                                            number_match.group(1)
                                        )
                                        break

    def _process_constituents_table(self, table, result):
        """Extract top constituents from a table"""
        # Check if this looks like a constituents table
        name_col = None
        weight_col = None

        # Try to identify the name and weight columns
        for col in table.columns:
            col_str = str(col).lower()
            if name_col is None and any(
                term in col_str
                for term in [
                    "name",
                    "company",
                    "security",
                    "stock",
                    "constituent",
                    "scrip",
                ]
            ):
                name_col = col
            elif weight_col is None and any(
                term in col_str for term in ["weight", "%", "allocation", "percent"]
            ):
                weight_col = col

        # If we couldn't identify columns by header, assume first column is name and second is weight
        if name_col is None and table.shape[1] >= 1:
            name_col = table.columns[0]
        if weight_col is None and table.shape[1] >= 2:
            weight_col = table.columns[1]

        # Extract constituents if we have the needed columns
        if name_col is not None and weight_col is not None:
            for _, row in table.iterrows():
                name = str(row[name_col]).strip()
                weight = str(row[weight_col]).strip()

                # Skip if name is empty or just header text
                if not name or name.lower() in [
                    "name",
                    "company",
                    "constituent",
                    "security",
                    "stock",
                    "scrip",
                ]:
                    continue

                # Extract the numeric part of the weight
                weight_match = re.search(r"([-+]?\d*\.?\d+)", weight)
                if weight_match:
                    weight = weight_match.group(1)

                    # Clean up company name
                    name = name.replace("Ltd", "Ltd.")
                    name = " ".join(word.capitalize() for word in name.split())
                    name = re.sub(r"Ltd$", "Ltd.", name)

                    # Only add if weight seems reasonable (0-100%)
                    try:
                        weight_float = float(weight)
                        if 0 <= weight_float <= 100:
                            result["top_constituents"].append(
                                {"name": name, "weight": weight}
                            )
                    except ValueError:
                        # Skip if weight can't be converted to float
                        pass

    def _process_sector_table(self, table, result):
        """Extract sector representation from a table"""
        # Try to identify the sector and weight columns
        sector_col = None
        weight_col = None

        # Try to identify the sector and weight columns
        for col in table.columns:
            col_str = str(col).lower()
            if sector_col is None and any(
                term in col_str
                for term in ["sector", "industry", "allocation", "exposure"]
            ):
                sector_col = col
            elif weight_col is None and any(
                term in col_str for term in ["weight", "%", "allocation", "percent"]
            ):
                weight_col = col

        # If we couldn't identify columns by header, assume first column is sector and second is weight
        if sector_col is None and table.shape[1] >= 1:
            sector_col = table.columns[0]
        if weight_col is None and table.shape[1] >= 2:
            weight_col = table.columns[1]

        # Extract sectors if we have the needed columns
        if sector_col is not None and weight_col is not None:
            for _, row in table.iterrows():
                sector = str(row[sector_col]).strip()
                weight = str(row[weight_col]).strip()

                # Skip if sector is empty or just header text
                if not sector or sector.lower() in [
                    "sector",
                    "industry",
                    "allocation",
                    "exposure",
                ]:
                    continue

                # Extract the numeric part of the weight
                weight_match = re.search(r"([-+]?\d*\.?\d+)", weight)
                if weight_match:
                    weight = weight_match.group(1)

                    # Clean up sector name
                    sector = " ".join(word.capitalize() for word in sector.split())

                    # Only add if weight seems reasonable (0-100%)
                    try:
                        weight_float = float(weight)
                        if 0 <= weight_float <= 100:
                            result["sector_representation"][sector] = weight
                    except ValueError:
                        # Skip if weight can't be converted to float
                        pass

    def _process_portfolio_table(self, table, result):
        """Extract portfolio characteristics from a table"""
        # Look for key-value pairs in the table
        key_value_map = {
            "methodology": "Methodology",
            "no. of constituent": "No. of Constituents",
            "launch date": "Launch Date",
            "base date": "Base Date",
            "base value": "Base Value",
            "calculation frequency": "Calculation Frequency",
            "index rebalancing": "Index Rebalancing",
        }

        # Examine each row to find these key-value pairs
        for _, row in table.iterrows():
            if len(row) < 2:  # Skip rows with only one cell
                continue

            # Check first cell for key
            key_cell = str(row.iloc[0]).lower().strip()

            for key_pattern, result_key in key_value_map.items():
                if key_pattern in key_cell:
                    # Found a match, get the value from the next cell
                    value = str(row.iloc[1]).strip()
                    if value:  # Only add non-empty values
                        result["portfolio_characteristics"][result_key] = value
                    break

    def _process_stats_table(self, table, result):
        """Extract statistics and fundamentals from a table"""
        logger.debug("Processing statistics table")

        # Check for specific headers that would indicate it's a statistics table
        # First, convert all column names and values to lowercase strings for matching
        col_names_lower = [str(col).lower() if col else "" for col in table.columns]

        # Check if this looks like the statistics table in the factsheet image
        if any("statistics" in col for col in col_names_lower) or any(
            "std" in col for col in col_names_lower
        ):
            logger.debug("Found statistics table with standard format")

            # Try to find the key statistics rows (Std. Deviation, Beta, Correlation)
            for idx, row in table.iterrows():
                row_as_str = [str(cell).lower().strip() if cell else "" for cell in row]

                # Check if this row has the stat name
                stat_name = None
                if any("std" in cell or "deviation" in cell for cell in row_as_str):
                    stat_name = "Std. Deviation"
                elif any("beta" in cell for cell in row_as_str):
                    stat_name = "Beta (NIFTY 50)"
                elif any("correlation" in cell for cell in row_as_str):
                    stat_name = "Correlation (NIFTY 50)"

                if stat_name:
                    # Find the columns with the values
                    period_cols = {}
                    for col_idx, col_name in enumerate(table.columns):
                        col_name_lower = str(col_name).lower().strip()

                        # Map column names to standardized period names
                        if "1 year" in col_name_lower:
                            period_cols["1 Year"] = col_idx
                        elif "5 years" in col_name_lower:
                            period_cols["5 Years"] = col_idx
                        elif "since inception" in col_name_lower:
                            period_cols["Since Inception"] = col_idx

                    # Extract values for each period
                    for period, col_idx in period_cols.items():
                        value = row.iloc[col_idx]
                        if value and str(value).strip():
                            # Extract number including decimal points
                            number_match = re.search(
                                r"([-+]?[0-9]*\.?[0-9]+)", str(value)
                            )
                            if number_match:
                                result["statistics"][f"{stat_name} ({period})"] = (
                                    number_match.group(1)
                                )

        # Alternative approach if the standard approach doesn't work
        else:
            logger.debug("Using alternative approach for statistics table")

            # Keys to look for in statistics and fundamentals
            stats_keys = {
                "Std. Deviation": [
                    "std",
                    "standard deviation",
                    "std dev",
                    "std. dev",
                    "deviation",
                    "volatility",
                ],
                "Beta (NIFTY 50)": ["beta", "beta (nifty", "beta nifty"],
                "Correlation (NIFTY 50)": [
                    "correlation",
                    "correlation (nifty",
                    "correlation nifty",
                ],
                "P/E": ["p/e", "pe ratio", "price to earnings", "price/earnings"],
                "P/B": ["p/b", "pb ratio", "price to book", "price/book"],
                "Dividend Yield": ["dividend", "dividend yield", "yield"],
            }

            # Check each row for stats
            for _, row in table.iterrows():
                # Convert row to string for easier pattern matching
                row_str = row.to_string().lower()

                # Check each stat key
                for stat_key, patterns in stats_keys.items():
                    if any(pattern in row_str for pattern in patterns):
                        # Found a match, extract the value
                        for cell in row:
                            if cell and isinstance(cell, (str, int, float)):
                                cell_str = str(cell)
                                number_match = re.search(
                                    r"([-+]?[0-9]*\.?[0-9]+)", cell_str
                                )
                                if number_match:
                                    value = number_match.group(1)

                                    # Add to the appropriate category
                                    if stat_key in [
                                        "Std. Deviation",
                                        "Beta (NIFTY 50)",
                                        "Correlation (NIFTY 50)",
                                    ]:
                                        result["statistics"][stat_key] = value
                                    elif stat_key in ["P/E", "P/B", "Dividend Yield"]:
                                        result["fundamentals"][stat_key] = value

    def _extract_from_text(self, text, result):
        """Extract data from text using regex patterns"""
        # Patterns for portfolio characteristics
        portfolio_patterns = [
            (r"methodology[\s:]+([^0-9\n]{3,50})", "Methodology"),
            (r"no\.?\s+of\s+constituents[\s:]+([0-9]+)", "No. of Constituents"),
            (r"launch\s+date[\s:]+([A-Za-z0-9,\s]+?\d{4})", "Launch Date"),
            (r"inception\s+date[\s:]+([A-Za-z0-9,\s]+?\d{4})", "Launch Date"),
            (r"base\s+date[\s:]+([A-Za-z0-9,\s]+?\d{4})", "Base Date"),
            (r"base\s+value[\s:]+([0-9,\.]+)", "Base Value"),
            (r"calculation\s+frequency[\s:]+([^\n]+)", "Calculation Frequency"),
            (r"index\s+rebalancing[\s:]+([^\n]+)", "Index Rebalancing"),
        ]

        # Extract portfolio characteristics from text
        for pattern, key in portfolio_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["portfolio_characteristics"][key] = match.group(1).strip()

        # If returns are still missing, try to extract them from text
        if not result["returns"]:
            returns_patterns = [
                # 1 Year returns
                (r"1[\s-]*year(?:\sreturn)?[\s:]*([0-9\.-]+)", "1 Year"),
                (r"1[\s-]*yr(?:\sreturn)?[\s:]*([0-9\.-]+)", "1 Year"),
                # 5 Year returns
                (r"5[\s-]*years?(?:\sreturn)?[\s:]*([0-9\.-]+)", "5 Years"),
                (r"5[\s-]*yrs?(?:\sreturn)?[\s:]*([0-9\.-]+)", "5 Years"),
                # YTD Returns
                (r"ytd(?:\sreturn)?[\s:]*([0-9\.-]+)", "YTD"),
                (r"year[\s-]*to[\s-]*date(?:\sreturn)?[\s:]*([0-9\.-]+)", "YTD"),
                # Price Return
                (r"price\s+return[\s:]*([0-9\.-]+)", "Price Return"),
                # Total Return
                (r"total\s+return[\s:]*([0-9\.-]+)", "Total Return"),
            ]

            for pattern, key in returns_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result["returns"][key] = match.group(1).strip()

        # If top_constituents are still missing, try to extract them from text
        if not result["top_constituents"]:
            # Look for patterns like "Company Name Ltd. 10.04"
            generic_pattern = r"([A-Za-z][A-Za-z\s\.&']+(?:Ltd\.?|Limited|Corporation|Corp\.|India))[\s:]+([0-9\.]+)"

            # First try to find a section that contains top constituents
            constituent_sections = re.findall(
                r"(?:top constituents|top holdings|top stocks|major constituents).*?(?:\n(?:.*\n){1,30})",
                text,
                re.IGNORECASE,
            )
            search_text = (
                "\n".join(constituent_sections) if constituent_sections else text
            )

            matches = re.finditer(generic_pattern, search_text, re.IGNORECASE)

            for match in matches:
                if len(match.groups()) >= 2:
                    company_name = match.group(1).strip()
                    weight = match.group(2).strip()

                    # Filter out false positives
                    try:
                        weight_float = float(weight)
                        if len(company_name) > 3 and 0 <= weight_float <= 100:
                            # Clean up company name - capitalize words properly
                            company_name = " ".join(
                                word.capitalize() for word in company_name.split()
                            )
                            # Fix common company name patterns
                            company_name = re.sub(r"Ltd$", "Ltd.", company_name)
                            result["top_constituents"].append(
                                {"name": company_name, "weight": weight}
                            )
                    except ValueError:
                        pass

    def _post_process_results(self, result):
        """Ensure data quality and fix any inconsistencies"""
        # Clean up portfolio characteristics
        for key in result.get("portfolio_characteristics", {}):
            value = result["portfolio_characteristics"][key]

            # Remove any unnecessary double dots in company/index names
            if (
                ".." in value
                and key != "Index Rebalancing"
                and key != "Calculation Frequency"
            ):
                result["portfolio_characteristics"][key] = value.replace("..", ".")

        # Ensure returns are valid numbers (can be negative)
        for key in result.get("returns", {}):
            value = result["returns"][key]
            try:
                # Try to convert to float to validate
                float_value = float(value)
                # Keep the original string format
                result["returns"][key] = value
            except (ValueError, TypeError):
                # If it's not a valid number, try to extract just the number part
                number_match = re.search(r"([-+]?\d*\.?\d+)", value)
                if number_match:
                    result["returns"][key] = number_match.group(1)
                else:
                    # If we can't extract a number, remove this entry
                    result["returns"].pop(key, None)

        # Ensure no duplicate constituents
        if result["top_constituents"]:
            unique_constituents = {}
            for constituent in result["top_constituents"]:
                name = constituent["name"]
                weight = constituent["weight"]

                # Use the highest weight if there are duplicates
                if name not in unique_constituents or float(weight) > float(
                    unique_constituents[name]["weight"]
                ):
                    unique_constituents[name] = {"name": name, "weight": weight}

            # Convert back to list and sort by weight (descending)
            result["top_constituents"] = sorted(
                list(unique_constituents.values()),
                key=lambda x: float(x["weight"]),
                reverse=True,
            )

            # Only keep the top constituents (limit to 10 if there are more)
            if len(result["top_constituents"]) > 10:
                result["top_constituents"] = result["top_constituents"][:10]

        # Move sectors mistakenly identified as constituents to sector_representation
        if result["top_constituents"]:
            common_sectors = [
                "Financial Services",
                "Information Technology",
                "Consumer Goods",
                "Healthcare",
                "Energy",
                "Automobile",
                "Metals",
                "Construction",
                "Telecom",
                "Media",
                "Services",
                "Consumer Durables",
                "Chemicals",
                "Power",
                "Oil",
                "Gas",
                "Mining",
                "Realty",
            ]

            sectors_to_move = []
            for i, constituent in enumerate(result["top_constituents"]):
                # Check if this looks like a sector rather than a company
                name = constituent["name"]
                if (
                    any(sector.lower() in name.lower() for sector in common_sectors)
                    and "ltd" not in name.lower()
                ):
                    if not any(
                        company_word in name.lower()
                        for company_word in ["inc", "corp", "company", "limited"]
                    ):
                        # This is likely a sector
                        result["sector_representation"][name] = constituent["weight"]
                        sectors_to_move.append(i)

            # Remove sectors from constituents (backwards to avoid index issues)
            for i in sorted(sectors_to_move, reverse=True):
                result["top_constituents"].pop(i)

    def save_to_json(self, parsed_data, output_path=None):
        """Save parsed data to JSON"""
        if output_path is None:
            base_filename = os.path.splitext(parsed_data["file_name"])[0]
            output_path = os.path.join(self.output_folder, f"{base_filename}.json")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved parsed data to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving to JSON: {str(e)}")
            return None

    def process_all_pdfs(self, pdf_folder, max_workers=4):
        """Process all PDFs in the given folder"""
        # Find all PDF files
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_folder}")
            return []

        logger.debug(f"Found {len(pdf_files)} PDF files to process with pdfplumber")

        processed_files = []

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for parsing
            future_to_file = {
                executor.submit(self.parse_factsheet, str(pdf_file)): pdf_file
                for pdf_file in pdf_files
            }

            # Use tqdm to display progress
            for future in tqdm(
                concurrent.futures.as_completed(future_to_file),
                total=len(future_to_file),
                desc="Parsing PDFs",
            ):
                pdf_file = future_to_file[future]
                try:
                    result = future.result()
                    if result["success"]:
                        json_path = self.save_to_json(result)
                        if json_path:
                            processed_files.append(json_path)
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {str(e)}")

        logger.debug(
            f"Successfully processed {len(processed_files)} PDFs with pdfplumber"
        )
        return processed_files


def main():
    """Main function to download and parse NSE index factsheets"""
    parser = argparse.ArgumentParser(
        description="Download and parse NSE index factsheets"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download factsheets without parsing",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Only parse existing factsheets without downloading new ones",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing",
    )
    parser.add_argument(
        "--input",
        default="nifty_factsheets",
        help="Input folder with PDF files (for parse-only mode)",
    )
    parser.add_argument(
        "--output", default="parsed_factsheets", help="Output folder for JSON files"
    )

    args = parser.parse_args()

    # Download factsheets if needed
    if not args.parse_only:
        logger.info("Starting download of NSE index factsheets")
        downloader = NSEFactsheetDownloader(output_folder=args.input)
        downloaded_files = downloader.download_all_factsheets()
        logger.info(f"Downloaded {len(downloaded_files)} factsheets")

    # Parse factsheets if needed
    if not args.download_only:
        logger.info("Starting parsing of NSE index factsheets")
        parser = PDFFactsheetParser(output_folder=args.output)
        processed_files = parser.process_all_pdfs(args.input, max_workers=args.workers)
        logger.info(f"Processed {len(processed_files)} factsheets")

    logger.info("Process completed successfully")


if __name__ == "__main__":
    main()
