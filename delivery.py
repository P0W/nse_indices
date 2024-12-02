"""Indices analyzer"""

import logging
import csv
import os

from enum import Enum
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    # pylint: disable=line-too-long
    format="%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class NSEClient:
    """NSE Client to fetch ETF data"""

    def __init__(self):
        self.base_url = "https://www.nseindia.com/"
        # pylint: disable=line-too-long
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36",
            "Accept-Language": "en,gu;q=0.9,hi;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
        }
        self.cookies = None
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a session with the required headers"""
        session = requests.Session()
        # session.headers.update(self.headers)

        try:
            request = session.get(self.base_url, headers=self.headers, timeout=10)
            self.cookies = dict(request.cookies)
        except requests.exceptions.RequestException as e:
            logger.error("Error in creating session: %s", e)
            return None
        return session

    def _get_data(self, url: str, top_n: int = None) -> List[Dict[str, float]]:
        """Get the top N ETFs by traded quantity"""
        if not self.session:
            return []
        try:
            response = self.session.get(
                url, timeout=5, cookies=self.cookies, headers=self.headers
            )
        except requests.exceptions.RequestException as e:
            logger.error("Request failed for %s: %s", url, e)
            return []
        logger.info("Response Code %s", response.status_code)
        response.raise_for_status()
        data_response = response.json()
        scrip_list = []
        for scrip in data_response["data"]:
            try:
                if "meta" not in scrip:
                    continue
                if "ltP" not in scrip:
                    ltp = scrip["lastPrice"]
                else:
                    ltp = scrip["ltP"]
                if "per" not in scrip:
                    per = scrip["pChange"]
                else:
                    per = scrip["per"]
                if "trdVal" not in scrip:
                    trade_voume = scrip["totalTradedVolume"]
                else:
                    trade_voume = scrip["qty"]
                scrip_list.append(
                    {
                        "symbol": scrip["symbol"],
                        "companyName": scrip["meta"]["companyName"],
                        "qty": float(trade_voume),
                        "ltP": float(ltp),
                        "per": float(per),
                    }
                )
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error in processing %s: %s", scrip, e)
        scrip_list = sorted(scrip_list, key=lambda x: x["qty"], reverse=True)
        return scrip_list[:top_n] if top_n else scrip_list

    class Index(Enum):
        """NSE Indices"""

        NIFTY_50 = "NIFTY%2050"
        NIFTY_200 = "NIFTY%20200"
        NIFTY_500 = "NIFTY%20500"
        ETF = "ETF"

    def get_list(self, index_value: Index, top_n: int = 10) -> List[Dict[str, float]]:
        """Get the top N equities by traded quantity"""
        if "NIFTY" in index_value.value:
            url = f"https://www.nseindia.com/api/equity-stockIndices?index={index_value.value}"
        else:
            url = "https://www.nseindia.com/api/etf"
        return self._get_data(url, top_n)

    def fetch_delivery_to_traded_qty(
        self, scrip_symbol: str
    ) -> Tuple[str, Optional[float]]:
        """Fetch the delivery to traded quantity ratio for a given symbol"""
        url = f"https://www.nseindia.com/api/quote-equity?symbol={scrip_symbol}&section=trade_info"
        try:
            logging.info("Fetching delivery to traded quantity for %s", scrip_symbol)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            json_data = response.json()
            delivery = json_data["securityWiseDP"]["deliveryToTradedQuantity"]
            return scrip_symbol, float(delivery)
        except requests.exceptions.RequestException as e:
            logger.error("Request failed for %s: %s", scrip_symbol, e)
            return scrip_symbol, None

    def get_quotes(
        self, index_value: Index, top_n: int = None
    ) -> Dict[str, Dict[str, float]]:
        """Get the top N scrips by traded quantity along with delivery to traded quantity ratio"""
        scrip_list = self.get_list(index_value, top_n)
        symbols = [scrip["symbol"] for scrip in scrip_list]
        results = {}

        with ThreadPoolExecutor() as executor:
            future_to_symbol = {
                executor.submit(self.fetch_delivery_to_traded_qty, symbol): symbol
                for symbol in symbols
            }
            for future in as_completed(future_to_symbol):
                scrip_symbol = future_to_symbol[future]
                try:
                    scrip_symbol, delivery = future.result()
                    idx = symbols.index(scrip_symbol)
                    results[scrip_symbol] = {
                        "delivery": delivery,
                        "ltp": scrip_list[idx]["ltP"],
                        "per": scrip_list[idx]["per"],
                    }
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("Request failed for %s: %s", scrip_symbol, e)
                    results[scrip_symbol] = None

        results = {
            symbol: data
            for symbol, data in results.items()
            if data is not None
            and data["delivery"] is not None
            and data["delivery"] > 0.0
        }
        results = dict(
            sorted(
                results.items(),
                key=lambda x: x[1]["delivery"],
                reverse=True,
            )
        )

        return results


def main(base_folder: str):
    """Main function to fetch the top 25 stocks by delivery percentage"""
    nse_client = NSEClient()
    # scrip_quotes = nse_client.get_quotes(NSEClient.Index.ETF, 20)
    # logger.info(json.dumps(scrip_quotes, indent=2))
    csv_header = ["Symbol", "LTP", "Change(%)", "Delivery(%)"]
    ## create folder delivery
    sub_folder = os.path.join(base_folder, "delivery_top_25")
    os.makedirs(sub_folder, exist_ok=True)
    for index in NSEClient.Index:
        ## Write to csv file
        file_name = index.value.replace("%20", "_")
        full_path = os.path.join(sub_folder, f"{file_name}.csv")
        logger.info("Top 25 stocks in %s", file_name)
        scrip_quotes = nse_client.get_quotes(index, 20)
        with open(full_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)
            for symbol, data in scrip_quotes.items():
                writer.writerow([symbol, data["ltp"], data["per"], data["delivery"]])


if __name__ == "__main__":
    main("data")
