"""Builds up the NSE Thematic and Startegy Indices and analyzes same"""

## Author: Prashant Srivastava
## Dated: August 12th, 2024

import json
import csv
import os
import argparse
import concurrent.futures
import pathlib
import logging
from typing import Dict
from functools import lru_cache

import requests
import tqdm
from bs4 import BeautifulSoup

import delivery

## set the logging level
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    # pylint: disable=line-too-long
    format="%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

file_list = [
    "ind_niftymidcap150momentum50_list",
    "ind_niftymicrocap250_list",
    "ind_niftyindiadefence_list",
    "ind_niftysmelist",
]


def get_nse_constituents(
    file_name: str = "ind_niftysmelist", root_dir: str = "."
) -> Dict[str, str]:
    """Download the NSE SME Emerge Constituents and return the symbols and company names"""
    ## check if already downloaded
    full_file_path = os.path.join(root_dir, file_name)
    ## add the extension
    full_file_path += ".csv"
    if pathlib.Path(full_file_path).exists():
        ## remove the file if older than 1 day
        os.remove(full_file_path)
    url = f"https://niftyindices.com/IndexConstituent/{file_name}.csv"
    ## NSE headers
    ## pylint: disable=line-too-long
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*,q=0.8,application/signed-exchange;v=b3;q=0.9",
    }
    try:
        r = requests.get(url, timeout=10, headers=headers)
    except requests.exceptions.RequestException as e:
        logger.error("Request failed for %s: %s", url, e)
        return {}
    if r.status_code != 200:
        logger.info("Error: %d : (%s)", r.status_code, url)
        return {}
    with open(full_file_path, "wb") as f:
        f.write(r.content)
    logger.info("Downloaded %s", full_file_path)
    ## headers = [Company Name,Industry,Symbol,Series,ISIN Code]
    ## get only the symbols and Company Name in a dictionary
    csv_header = ["Company Name", "Industry", "Symbol", "Series", "ISIN Code"]
    symbols = {}
    with open(full_file_path, "r", encoding="utf-8") as f:
        csv_reader = csv.DictReader(f, fieldnames=csv_header)
        next(csv_reader)  ## skip the header
        for row in csv_reader:
            symbols[row["Symbol"]] = row["Company Name"]
    return symbols


@lru_cache(maxsize=1024)
def get_financials(symbol: str) -> Dict[str, float]:
    """Get the financials for the given symbol"""
    url = f"https://ticker.finology.in/company/{symbol}"
    ratio_dict = {}
    try:
        res = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as e:
        logger.error("Request failed for %s: %s", url, e)
        return ratio_dict
    if res.status_code != 200:
        logger.info("Error: %d : (%s)", res.status_code, symbol)
        return ratio_dict
    parser = BeautifulSoup(res.text, "html.parser")
    section = parser.select("#mainContent_updAddRatios")

    for div in section:
        for small, p in zip(div.select("small"), div.select("p")):
            ## strip off the whitespace and newlines
            val = (
                p.text.replace("\n", "")
                .replace("Cr.", "")
                .replace("₹", "")
                .replace(",", "")
                .replace("%", "")
                .replace("-", "")
                .strip()
            )
            try:
                ratio_dict[small.text.strip()] = float(val)
            except ValueError:
                ratio_dict[small.text.strip()] = -999.999
    if "CASH" in ratio_dict and "DEBT" in ratio_dict:
        ratio_dict["Cash/Debt"] = float(
            round(ratio_dict["CASH"] - ratio_dict["DEBT"], 2)
        )

    return ratio_dict


def parse_finology(
    symbols_dict: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Parse the finology website for the financial ratios"""
    max_cores = os.cpu_count()
    logger.debug("Max cores: %d", max_cores)
    financial_ratio = {}
    batch_size = 15
    symbols_list = list(symbols_dict.keys())
    max_len = len(symbols_list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_cores) as executor:
        for i in tqdm.tqdm(
            range(0, max_len, batch_size),
            desc="Processing symbols in batches",
        ):
            chunk = symbols_list[i : i + batch_size]
            futures = {
                executor.submit(get_financials, company_symbol): company_symbol
                for company_symbol in chunk
            }
            for future in concurrent.futures.as_completed(futures):
                company_symbol = futures[future]
                try:
                    financial_ratio[company_symbol] = future.result()
                    ## add company name
                    financial_ratio[company_symbol]["Company Name"] = symbols_dict[
                        company_symbol
                    ]
                except Exception as e:  ##pylint: disable=broad-exception-caught
                    logger.info("Error processing %s:%s", company_symbol, e)
    return financial_ratio


# Custom sorting key function
def sorting_key(company_data: Dict[str, float]) -> tuple:
    """Custom sorting key function"""
    return (
        -company_data.get("Market Cap", 0),  # Largest to smallest
        -company_data.get("Cash/Debt", 0),  # Largest to smallest
        company_data.get("P/B", float("inf")),  # Smallest to largest
        -company_data.get("Promoter Holding", 0),  # Largest to smallest
        -company_data.get("ROCE", 0),  # Largest to smallest
        -company_data.get("ROE", 0),  # Largest to smallest
        -company_data.get("Profit Growth", 0),  # Largest to smallest
        -company_data.get("EPS (TTM)", 0),  # Largest to smallest
    )


def build_table(index_file: str, root_dir: str = ".", generate_json: bool = False):
    """Build the table for the given index file"""
    symbols_dict = get_nse_constituents(index_file, root_dir)
    ## call the get_financials for each symbol,
    financial_ratio = parse_finology(symbols_dict)
    total_market_cap = sum(
        company_data.get("Market Cap", 0) for company_data in financial_ratio.values()
    )
    # Add weightage to the financial_ratio based on Total Market Cap
    for company_symbol, company_data in financial_ratio.items():
        company_data["Weightage"] = round(
            (company_data.get("Market Cap", 0) / total_market_cap) * 100.0, 2
        )
    # Sort the financial_ratio dictionary
    financial_ratio = dict(
        sorted(financial_ratio.items(), key=lambda item: sorting_key(item[1]))
    )
    ## write to a json file suffixed with financials
    if generate_json:
        financial_ratio_file = os.path.join(root_dir, index_file + "_financials.json")
        with open(financial_ratio_file, "w", encoding="utf-8") as file_handle:
            json.dump(financial_ratio, file_handle, indent=2)

    ## write to a csv file
    # Access the first key of the financial_ratio dictionary
    first_key = list(financial_ratio.keys())[0]
    header_fields = financial_ratio[first_key].keys()
    ## sort the header fields
    header_fields = sorted(header_fields)

    ## write to a csv file suffixed with financials
    financial_ratio_file = os.path.join(root_dir, index_file + "_financials.csv")
    with open(financial_ratio_file, "w", newline="", encoding="utf-8") as file_handle:
        csv_writer = csv.writer(file_handle)
        ## write all the headers
        csv_writer.writerow(["Company Symbol"] + list(header_fields))
        for company_symbol, ratios in financial_ratio.items():
            csv_writer.writerow(
                [company_symbol] + [ratios.get(field, 0) for field in header_fields]
            )

    logger.info("Done")


arg_parser = argparse.ArgumentParser(description="NSE Indicies analyzer")

arg_parser.add_argument(
    "-d", "--data_dir", type=str, help="folder where to output", required=True
)
arg_parser.add_argument(
    "-g",
    "--generate_json",
    action="store_true",
    help="Generate JSON files, along with csv",
)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    if not pathlib.Path(args.data_dir).exists():
        os.makedirs(args.data_dir)
    try:
        delivery.main(args.data_dir)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error in delivery.main: %s", e)
    for file in file_list:
        try:
            build_table(
                index_file=file,
                root_dir=args.data_dir,
                generate_json=args.generate_json,
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error in build_table: %s", e)
