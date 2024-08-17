"""
Download N shares from yahoo finance for last 5 years daily data
    create a basket of N shares with 50% weightage each and set base/starting value as 100
    create a class Charter with following methods
    calculate the daily returns for the basket
    calculate the cumulative returns for the basket
    calculate the annualized returns for the basket
    calculate the annualized volatility for the basket
    calculate the sharpe ratio for the basket
    calculate the sortino ratio for the basket
    calculate the maximum drawdown for the basket
    calculate the calmar ratio for the basket
    calculate the beta for the basket
    calculate the alpha for the basket
    calculate the information ratio for the basket
    calculate the treynor ratio for the basket
    calculate the tracking error for the basket
    calculate the jensen's alpha for the basket
    calculate the r-squared for the basket
    calculate the standard deviation for the basket
    calculate the mean for the basket
    calculate the skewness for the basket
    calculate the kurtosis for the basket
"""

## Author: Prashant Srivastava
## Dated: August 17th, 2024

import os
import datetime
import logging
import json
from typing import List
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

## Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## pylint: disable=logging-fstring-interpolation


## pylint: disable=too-many-public-methods
class Charter:
    """
    Charter class to calculate various metrics for a basket of stocks
    """

    def __init__(self, stocks: List[str], start_date, end_date, download_folder: str):
        """Initialize the Charter class with the given stocks, start date, end date,
        and download directory"""
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.download_dir = download_folder
        self.data = pd.DataFrame()

    def get_data(self):
        """Download the stock data from Yahoo Finance"""
        stock_data_list = []

        for stock in self.stocks:
            file_path = os.path.join(self.download_dir, f"{stock}.csv")

            if not os.path.exists(file_path):
                data = yf.download(stock, start=self.start_date, end=self.end_date)
                data.to_csv(file_path)

            stock_data = pd.read_csv(file_path, index_col="Date")
            stock_data = stock_data.drop(
                ["Open", "High", "Low", "Close", "Volume"], axis=1
            )
            stock_data.columns = [stock]
            stock_data[stock] = 100 * stock_data[stock] / stock_data[stock].iloc[0]
            stock_data_list.append(stock_data)

        self.data = pd.concat(stock_data_list, axis=1)

    def calculate_basket_daily_returns(self):
        """Calculate the daily returns for the basket"""
        self.data["Basket"] = self.data.mean(axis=1)
        self.data["Basket Daily Returns"] = self.data["Basket"].pct_change()
        self.data = self.data.dropna()

    def calculate_basket_cumulative_returns(self):
        """Calculate the cumulative returns for the basket"""
        self.data["Basket Cumulative Returns"] = (
            1 + self.data["Basket Daily Returns"]
        ).cumprod()

    def calculate_basket_annualized_returns(self):
        """Calculate the annualized returns for the basket"""
        self.data["Basket Annualized Returns"] = (
            self.data["Basket Cumulative Returns"].iloc[-1] ** (252 / len(self.data))
        ) - 1

    def calculate_basket_annualized_volatility(self):
        """Calculate the annualized volatility for the basket"""
        self.data["Basket Annualized Volatility"] = self.data[
            "Basket Daily Returns"
        ].std() * (252**0.5)

    def calculate_basket_sharpe_ratio(self):
        """Calculate the Sharpe Ratio for the basket"""
        self.data["Basket Sharpe Ratio"] = (
            self.data["Basket Annualized Returns"]
            / self.data["Basket Annualized Volatility"]
        )

    def calculate_basket_sortino_ratio(self):
        """Calculate the Sortino Ratio for the basket"""
        self.data["Basket Sortino Ratio"] = (
            self.data["Basket Annualized Returns"]
            / self.data["Basket Daily Returns"][
                self.data["Basket Daily Returns"] < 0
            ].std()
        )

    def calculate_basket_maximum_drawdown(self):
        """Calculate the maximum drawdown for the basket"""
        self.data["Basket Maximum Drawdown"] = (
            1
            - self.data["Basket Cumulative Returns"]
            / self.data["Basket Cumulative Returns"].cummax()
        ).max()

    def calculate_basket_calmar_ratio(self):
        """Calculate the Calmar Ratio for the basket"""
        self.data["Basket Calmar Ratio"] = (
            self.data["Basket Annualized Returns"]
            / self.data["Basket Maximum Drawdown"]
        )

    def calculate_basket_beta(self):
        """Calculate the beta for the basket"""
        betas = []
        for stock in self.stocks:
            cov = self.data["Basket Daily Returns"].cov(self.data[stock])
            var = self.data[stock].var()
            beta = cov / var
            self.data[f"{stock} Beta"] = beta
            betas.append(beta)

        # Calculate the average beta for the basket
        self.data["Basket Beta"] = sum(betas) / len(betas)

    def calculate_basket_alpha(self):
        """Calculate the alpha for the basket"""
        # Calculate the average daily return for all stocks
        avg_daily_return = self.data[self.stocks].pct_change().mean(axis=1).mean()

        # Calculate the basket alpha
        self.data["Basket Alpha"] = self.data["Basket Annualized Returns"] - (
            self.data["Basket Beta"] * avg_daily_return
        )

    def calculate_basket_information_ratio(self):
        """Calculate the Information Ratio for the basket"""
        ## for N stocks basket
        self.data["Basket Information Ratio"] = (
            self.data["Basket Alpha"] / self.data["Basket Daily Returns"].std()
        )

    def calculate_basket_treynor_ratio(self):
        """Calculate the Treynor Ratio for the basket"""
        self.data["Basket Treynor Ratio"] = (
            self.data["Basket Annualized Returns"] / self.data["Basket Beta"]
        )

    def calculate_basket_tracking_error(self):
        """Calculate the tracking error for the basket"""
        self.data["Basket Tracking Error"] = (
            self.data[self.stocks].pct_change().std(axis=1)
        )

    def calculate_basket_jensens_alpha(self):
        """Calculate the Jensen's Alpha for the basket"""
        ## for N stocks basket
        self.data["Basket Jensens Alpha"] = self.data["Basket Annualized Returns"] - (
            self.data["Basket Beta"]
            * (self.data[self.stocks].pct_change().mean(axis=1).mean())
        )

    def calculate_basket_r_squared(self):
        """Calculate the R-squared value for the basket"""
        ## for N stocks basket
        self.data["Basket R-Squared"] = (
            self.data["Basket Daily Returns"].corr(
                self.data[self.stocks].pct_change().mean(axis=1)
            )
            ** 2
        )

    def calculate_basket_standard_deviation(self):
        """Calculate the standard deviation of the basket daily returns"""
        self.data["Basket Standard Deviation"] = self.data["Basket Daily Returns"].std()

    def calculate_basket_mean(self):
        """Calculate the mean of the basket daily returns"""
        self.data["Basket Mean"] = self.data["Basket Daily Returns"].mean()

    def calculate_basket_skewness(self):
        """Calculate the skewness of the basket daily returns"""
        self.data["Basket Skewness"] = self.data["Basket Daily Returns"].skew()

    def calculate_basket_kurtosis(self):
        """Calculate the kurtosis of the basket daily returns"""
        self.data["Basket Kurtosis"] = self.data["Basket Daily Returns"].kurtosis()

    def calculate_all(self):
        """Calculate all the metrics for the basket"""
        self.get_data()
        self.calculate_basket_daily_returns()
        self.calculate_basket_cumulative_returns()
        self.calculate_basket_annualized_returns()
        self.calculate_basket_annualized_volatility()
        self.calculate_basket_sharpe_ratio()
        self.calculate_basket_sortino_ratio()
        self.calculate_basket_maximum_drawdown()
        self.calculate_basket_calmar_ratio()
        self.calculate_basket_beta()
        self.calculate_basket_alpha()
        self.calculate_basket_information_ratio()
        self.calculate_basket_treynor_ratio()
        self.calculate_basket_tracking_error()
        self.calculate_basket_jensens_alpha()
        self.calculate_basket_r_squared()
        self.calculate_basket_standard_deviation()
        self.calculate_basket_mean()
        self.calculate_basket_skewness()
        self.calculate_basket_kurtosis()

    def display_periodic_cagr(self, periods: List[int] = None):
        """Display the Compound Annual Growth Rate (CAGR) for the last N years"""
        # Ensure the 'Date' column is in datetime format
        self.data.index = pd.to_datetime(self.data.index)

        if periods is None:
            periods = [1, 2, 3, 4, 5]

        # Get the end value (most recent value)
        end_value = self.data["Basket Cumulative Returns"].iloc[-1]
        end_date = self.data.index[-1].strftime("%Y-%m-%d")

        cagr = []
        period_labels = []

        for period in periods:
            # Calculate the start date for the period

            start_date = self.data.index[-1] - pd.DateOffset(years=period)
            if period == 5:
                logger.info(start_date)

            # Filter the data to find the closest date to the calculated start date
            filtered_data = self.data[self.data.index <= start_date]
            if not filtered_data.empty:
                start_value = filtered_data["Basket Cumulative Returns"].iloc[-1]
                actual_start_date = filtered_data.index[-1].strftime("%Y-%m-%d")

                # Calculate CAGR
                cagr_value = ((end_value / start_value) ** (1 / period)) - 1
                cagr.append(cagr_value * 100)
                period_labels.append((period, actual_start_date, end_date))
            else:
                # If no data is available for the period, append None
                logger.info("No data available for the period %d", period)
                ## find the period in lower 10 days
                for i in range(10):
                    start_date = (
                        self.data.index[-1]
                        - pd.DateOffset(years=period)
                        + pd.DateOffset(days=i)
                    )
                    filtered_data = self.data[self.data.index <= start_date]
                    if not filtered_data.empty:
                        start_value = filtered_data["Basket Cumulative Returns"].iloc[
                            -1
                        ]
                        actual_start_date = filtered_data.index[-1].strftime("%Y-%m-%d")
                        cagr_value = ((end_value / start_value) ** (1 / period)) - 1
                        cagr.append(cagr_value * 100)
                        period_labels.append((period, actual_start_date, end_date))
                        break

        logger.info("CAGR for the last N years:")
        for i, label in enumerate(period_labels):
            if cagr[i] is not None:
                logger.info(
                    f"Last {label[0]} year(s) ({label[1]} to {label[2]}): {cagr[i]:.2f}%"
                )
            else:
                logger.info(f"Last {label[0]} year(s): Data not available")

    def get_results(self):
        """Return the results as a DataFrame"""
        return self.data

    def save_results(self):
        """Save the results to a CSV file"""
        save_path = os.path.join(self.download_dir, "basket_results.csv")
        self.data.to_csv(save_path)

    def plot(self):
        """Plot the cumulative returns of the basket"""
        self.data["Basket Cumulative Returns"].plot()
        plt.show()

    def display_details(self):
        """Display the details of the basket"""
        ## pylint: disable=line-too-long
        overall_returns = self.data["Basket Annualized Returns"].iloc[-1]
        max_drawdown = self.data["Basket Maximum Drawdown"].iloc[-1]
        basket_overall_alpha = self.data["Basket Alpha"].iloc[-1]
        basket_overall_beta = self.data["Basket Beta"].iloc[-1]
        logger.info(f"Overall Basket Returns: {overall_returns:.2f}")
        logger.info(
            f"Overall Basket Volatility: {self.data['Basket Annualized Volatility'].iloc[-1]:.2f}"
        )
        logger.info(
            f"Overall Basket Sharpe Ratio: {self.data['Basket Sharpe Ratio'].iloc[-1]:.2f}"
        )
        logger.info(
            f"Overall Basket Sortino Ratio: {self.data['Basket Sortino Ratio'].iloc[-1]:.2f}"
        )
        logger.info(f"Overall Basket Max Drawdown: {max_drawdown:.2f}")
        logger.info(
            f"Overall Basket Calmar Ratio: {self.data['Basket Calmar Ratio'].iloc[-1]:.2f}"
        )
        logger.info(f"Overall Basket Alpha: {basket_overall_alpha:.2f}")
        logger.info(f"Overall Basket Beta: {basket_overall_beta:.2f}")
        logger.info(
            f"Overall Basket Information Ratio: {self.data['Basket Information Ratio'].iloc[-1]:.2f}"
        )
        logger.info(
            f"Overall Basket Treynor Ratio: {self.data['Basket Treynor Ratio'].iloc[-1]:.2f}"
        )
        logger.info(
            f"Overall Basket Tracking Error: {self.data['Basket Tracking Error'].iloc[-1]:.2f}"
        )
        logger.info(
            f"Overall Basket Jensens Alpha: {self.data['Basket Jensens Alpha'].iloc[-1]:.2f}"
        )
        logger.info(
            f"Overall Basket R-Squared: {self.data['Basket R-Squared'].iloc[-1]:.2f}"
        )
        logger.info(
            f"Overall Basket Standard Deviation: {self.data['Basket Standard Deviation'].iloc[-1]:.2f}"
        )
        logger.info(f"Overall Basket Mean: {self.data['Basket Mean'].iloc[-1]:.2f}")
        logger.info(
            f"Overall Basket Skewness: {self.data['Basket Skewness'].iloc[-1]:.2f}"
        )
        logger.info(
            f"Overall Basket Kurtosis: {self.data['Basket Kurtosis'].iloc[-1]:.2f}"
        )


def build_top_n_basket(json_indices_file, num_stocks):
    """Build a basket of top num_stocks stocks from the indices in the JSON file"""
    with open(json_indices_file, "r", encoding="utf-8") as file:
        indices = json.load(file)

    basket = []
    symbols = list(indices.keys())
    for index in symbols[:num_stocks]:
        basket.append(f"{index}.NS")

    return basket


if __name__ == "__main__":
    YEARS = 5
    now = datetime.datetime.now()
    years_ago = now - datetime.timedelta(days=365 * YEARS)
    download_dir = os.path.join(os.getcwd(), "chart_data")
    defence_basket = build_top_n_basket(
        json_indices_file="data/ind_niftyindiadefence_list_financials.json",
        num_stocks=5,
    )
    microcap_basket = build_top_n_basket(
        json_indices_file="data/ind_niftymicrocap250_list_financials.json", num_stocks=2
    )
    momentum50_basket = build_top_n_basket(
        json_indices_file="data/ind_niftymidcap150momentum50_list_financials.json",
        num_stocks=5,
    )
    logger.info(defence_basket)
    logger.info(microcap_basket)
    logger.info(momentum50_basket)
    charter = Charter(
        defence_basket,
        years_ago,
        now,
        download_dir,
    )
    charter.calculate_all()
    results = charter.get_results()
    charter.save_results()
    charter.display_periodic_cagr(periods=list(range(1, YEARS + 1)))
    charter.plot()
