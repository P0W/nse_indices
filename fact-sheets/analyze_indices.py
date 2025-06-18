#!/usr/bin/env python
"""
NSE Indices Analysis Tool

This script analyzes and visualizes performance metrics across all NSE index factsheets.
It provides comprehensive analysis of returns, risk metrics, correlations, and
generates visualizations to identify the best performing indices.

Key features:
- Load and parse JSON data from factsheet files
- Calculate risk-adjusted metrics (Sharpe ratios)
- Rank indices based on performance, risk, and risk-adjusted returns
- Generate visualizations for analysis
- Create investment recommendations
- Export analysis results to JSON for web display

The code has been optimized for:
- Clear organization and maintainability
- Type hints for better IDE support
- Comprehensive error handling
- Logging at appropriate levels
- PEP 8 compliance

Author: NSE Indices Analysis Team
Version: 1.1
Last Updated: 2025-06-18
"""

from __future__ import annotations
import json
import re
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set

# Data processing imports
import numpy as np
import pandas as pd

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Advanced analysis imports
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Machine learning imports
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("indices_analyzer.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class IndicesAnalyzer:
    """
    Analyzes NSE indices data from parsed factsheet JSON files to extract insights
    and generate visualizations for investment decisions.

    This class provides methods to:
    - Load and process data from JSON files
    - Calculate risk-adjusted metrics like Sharpe ratios
    - Generate visualizations of performance metrics
    - Perform cluster analysis to group similar indices
    - Generate investment recommendations
    """

    def __init__(
        self, json_dir: str = "parsed_factsheets", output_dir: str = "analysis_results"
    ) -> None:
        """
        Initialize the analyzer with directories for input and output

        Args:
            json_dir: Directory containing parsed factsheet JSON files
            output_dir: Directory to save analysis results and visualizations
        """
        self.json_dir = Path(json_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Dataframes to store consolidated data
        self.returns_df: Optional[pd.DataFrame] = None
        self.stats_df: Optional[pd.DataFrame] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.sharpe_ratios: Optional[pd.DataFrame] = None
        self.combined_df: Optional[pd.DataFrame] = None

        # Track the analysis date
        self.analysis_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Initializing IndicesAnalyzer with data from {self.json_dir}")

    def load_all_data(self) -> None:
        """
        Load and consolidate data from all JSON files into structured dataframes.

        This method:
        1. Reads all JSON files from the json_dir
        2. Extracts returns, statistics and metadata
        3. Converts string values to float where appropriate
        4. Creates pandas DataFrames for further analysis
        """
        logger.info("Loading data from all JSON files...")

        returns_data: List[Dict[str, Any]] = []
        stats_data: List[Dict[str, Any]] = []
        metadata: List[Dict[str, Any]] = []

        # Get all JSON files
        json_files = list(self.json_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in directory: {self.json_dir}")
            self._initialize_empty_dataframes()
            return

        logger.info(f"Found {len(json_files)} JSON files to analyze")

        for json_path in json_files:
            self._process_json_file(json_path, returns_data, stats_data, metadata)
        # Create dataframes
        self._create_dataframes_from_data(returns_data, stats_data, metadata)

        returns_count = (
            0
            if self.returns_df is None or self.returns_df.empty
            else len(self.returns_df)
        )
        stats_count = (
            0 if self.stats_df is None or self.stats_df.empty else len(self.stats_df)
        )
        logger.info(
            f"Loaded data for {returns_count} indices with returns and {stats_count} with statistics"
        )

    def _process_json_file(
        self,
        json_path: Path,
        returns_data: List[Dict[str, Any]],
        stats_data: List[Dict[str, Any]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Process a single JSON file and extract relevant data."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            index_name = self._get_index_name(json_path.stem, data)

            # Extract returns and convert to float
            if "returns" in data and data["returns"]:
                returns_row = {"Index": index_name}
                for metric, value in data["returns"].items():
                    returns_row[metric] = self._convert_to_float(
                        value, f"return {metric} for {index_name}"
                    )
                returns_data.append(returns_row)

            # Extract statistics and convert to float
            if "statistics" in data and data["statistics"]:
                stats_row = {"Index": index_name}
                for metric, value in data["statistics"].items():
                    stats_row[metric] = self._convert_to_float(
                        value, f"statistic {metric} for {index_name}"
                    )
                stats_data.append(stats_row)

            # Extract metadata
            meta_row = {"Index": index_name, "File": json_path.name}
            if "metadata" in data and data["metadata"]:
                meta_row.update({"Pages": data["metadata"].get("total_pages", 0)})
            metadata.append(meta_row)

        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")

    def _convert_to_float(self, value: Any, context: str) -> Optional[float]:
        """Convert a value to float, handling common formatting issues."""
        try:
            # Handle comma-separated numbers and convert to float
            if isinstance(value, str):
                clean_value = value.replace(",", "")
                return float(clean_value) if clean_value else None
            elif value is None:
                return None
            else:
                return float(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing {context}: {e}")
            return None

    def _create_dataframes_from_data(
        self,
        returns_data: List[Dict[str, Any]],
        stats_data: List[Dict[str, Any]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Create pandas DataFrames from the collected data."""
        if returns_data:
            self.returns_df = pd.DataFrame(returns_data).set_index("Index")
        else:
            logger.warning("No returns data found")
            self.returns_df = pd.DataFrame()

        if stats_data:
            self.stats_df = pd.DataFrame(stats_data).set_index("Index")
        else:
            logger.warning("No statistics data found")
            self.stats_df = pd.DataFrame()

        self.metadata_df = (
            pd.DataFrame(metadata).set_index("Index") if metadata else pd.DataFrame()
        )

    def _initialize_empty_dataframes(self) -> None:
        """Initialize empty DataFrames when no data is found."""
        self.returns_df = pd.DataFrame()
        self.stats_df = pd.DataFrame()
        self.metadata_df = pd.DataFrame()

    def _get_index_name(self, filename: str, data: Dict[str, Any]) -> str:
        """
        Extract a clean index name from filename or JSON data.

        Args:
            filename: The filename without extension
            data: The JSON data dictionary

        Returns:
            A clean, formatted index name
        """
        # Try to get name from metadata if available
        if "metadata" in data and data["metadata"] and "index_name" in data["metadata"]:
            name = data["metadata"]["index_name"]
        else:
            # Remove common prefixes and postfixes
            name = filename
            name = re.sub(r"^Factsheet_", "", name)
            name = re.sub(r"^ind_", "", name)
            name = re.sub(r"\.json$", "", name)
            name = re.sub(r"_Index$", "", name)
            name = re.sub(r"_Factsheet$", "", name)

            # Replace underscores with spaces
            name = name.replace("_", " ")

            # Clean up any double spaces
            name = re.sub(r"\s+", " ", name).strip()

        return name

    def calculate_risk_adjusted_metrics(self, risk_free_rate: float = 3.0) -> None:
        """
        Calculate risk-adjusted return metrics like Sharpe ratio.

        The Sharpe ratio measures the performance of an investment compared to a risk-free asset,
        after adjusting for its risk. It is calculated as:
        Sharpe Ratio = (Return - Risk Free Rate) / Standard Deviation

        Args:
            risk_free_rate: Annual risk-free rate in percentage (default: 3.0%)
        """
        logger.info("Calculating risk-adjusted metrics...")

        # Ensure we have both returns and statistics dataframes
        if (
            self.returns_df is None
            or self.stats_df is None
            or self.returns_df.empty
            or self.stats_df.empty
        ):
            logger.warning("Missing data for risk-adjusted metrics calculation")
            self.sharpe_ratios = pd.DataFrame()
            return

        # Create a dataframe for risk-adjusted metrics
        risk_adjusted = pd.DataFrame(index=self.returns_df.index)

        # Define the periods to analyze
        periods = ["1 Year", "5 Years", "Since Inception"]

        # For each period, calculate Sharpe ratio
        for period in periods:
            self._calculate_sharpe_for_period(risk_adjusted, period, risk_free_rate)

        self.sharpe_ratios = risk_adjusted
        logger.info(f"Calculated Sharpe ratios for {len(risk_adjusted)} indices")

    def _calculate_sharpe_for_period(
        self, df: pd.DataFrame, period: str, risk_free_rate: float
    ) -> None:
        """Calculate Sharpe ratio for a specific time period."""
        return_col = f"{period} Total Return"
        std_col = f"Std. Deviation ({period})"

        # Check if both columns exist in respective dataframes
        if return_col in self.returns_df.columns and std_col in self.stats_df.columns:
            # Join the data temporarily
            temp_df = pd.DataFrame(
                {
                    "Return": self.returns_df[return_col],
                    "StdDev": self.stats_df[std_col],
                }
            )

            # Filter out missing values
            temp_df = temp_df.dropna()

            # Avoid division by zero
            temp_df = temp_df[temp_df["StdDev"] > 0]

            if temp_df.empty:
                logger.warning(f"No valid data to calculate Sharpe ratio for {period}")
                return

            # Calculate Sharpe ratio: (Return - Risk_Free_Rate) / StdDev
            temp_df["Sharpe"] = (temp_df["Return"] - risk_free_rate) / temp_df["StdDev"]

            # Add to risk_adjusted dataframe
            df[f"Sharpe Ratio ({period})"] = temp_df["Sharpe"]

            logger.debug(
                f"Calculated Sharpe ratio for {period} with {len(temp_df)} valid indices"
            )

    def create_combined_dataset(self) -> None:
        """
        Create a combined dataset with returns, stats and risk metrics.

        This method merges data from different dataframes (returns, statistics, and Sharpe ratios)
        into a single DataFrame for easier analysis.

        The combined dataset is also saved as a JSON file for web display.
        """
        logger.info("Creating combined analysis dataset...")

        # Check if we have data to combine
        if self.returns_df is None or self.returns_df.empty:
            logger.warning("No returns data available for combined dataset")
            self.combined_df = pd.DataFrame()
            return

        # Create a copy of returns dataframe
        self.combined_df = self.returns_df.copy()

        # Add statistics if available
        if self.stats_df is not None and not self.stats_df.empty:
            self._merge_dataframe(self.combined_df, self.stats_df, "statistics")

        # Add Sharpe ratios if available
        if self.sharpe_ratios is not None and not self.sharpe_ratios.empty:
            self._merge_dataframe(self.combined_df, self.sharpe_ratios, "Sharpe ratios")

        # Add metadata if available
        if self.metadata_df is not None and not self.metadata_df.empty:
            self._merge_dataframe(self.combined_df, self.metadata_df, "metadata")

        logger.info(
            f"Created combined dataset with {len(self.combined_df.columns)} metrics"
        )

        # Save combined data to JSON for web display
        self._save_combined_data_to_json()

    def _merge_dataframe(
        self, target_df: pd.DataFrame, source_df: pd.DataFrame, description: str
    ) -> None:
        """Merge columns from source dataframe into target dataframe, avoiding duplicates."""
        columns_added = 0
        for col in source_df.columns:
            if col not in target_df:
                target_df[col] = source_df[col]
                columns_added += 1

        logger.debug(f"Added {columns_added} columns from {description}")

    def _save_combined_data_to_json(self) -> None:
        """Save the combined dataframe as JSON for web display."""
        if self.combined_df is not None and not self.combined_df.empty:
            try:
                # Convert to records format for easier JSON serialization
                data = self.combined_df.reset_index().to_dict(orient="records")

                # Save to JSON file
                output_file = self.output_dir / "combined_data.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                logger.info(f"Saved combined dataset to {output_file}")
            except Exception as e:
                logger.error(f"Error saving combined data to JSON: {e}")

    def rank_indices(self) -> None:
        """
        Rank indices based on various metrics.

        This method:
        1. Ranks indices based on returns (higher is better)
        2. Ranks indices based on risk metrics (lower is better)
        3. Ranks indices based on Sharpe ratios (higher is better)
        4. Calculates average ranks for each category
        5. Calculates an overall rank considering all categories
        """
        logger.info("Ranking indices based on performance metrics...")

        if self.combined_df is None or self.combined_df.empty:
            logger.warning("No combined data available for ranking")
            self.ranking_df = pd.DataFrame()
            return

        # Define metrics to use for ranking
        ranking_metrics = {
            "returns": [
                "1 Year Total Return",
                "5 Years Total Return",
                "Since Inception Total Return",
            ],
            "risk": [
                "Std. Deviation (1 Year)",
                "Std. Deviation (5 Years)",
                "Std. Deviation (Since Inception)",
            ],
            "sharpe": [
                "Sharpe Ratio (1 Year)",
                "Sharpe Ratio (5 Years)",
                "Sharpe Ratio (Since Inception)",
            ],
        }

        # Create a new dataframe to store ranks
        ranking_df = pd.DataFrame(index=self.combined_df.index)

        # Calculate ranks for each category
        category_ranks = []
        for category, metrics in ranking_metrics.items():
            category_rank = self._calculate_category_rank(category, metrics, ranking_df)
            if category_rank:
                category_ranks.append(category_rank)

        # Calculate overall rank if we have rankings for at least one category
        if category_ranks:
            ranking_df["Overall Rank"] = ranking_df[category_ranks].mean(axis=1)

        self.ranking_df = ranking_df
        logger.info(f"Completed ranking for {len(ranking_df)} indices")

    def _calculate_category_rank(
        self, category: str, metrics: List[str], ranking_df: pd.DataFrame
    ) -> Optional[str]:
        """Calculate ranks for a specific category of metrics."""
        # Filter metrics that exist in the dataframe
        available_metrics = [m for m in metrics if m in self.combined_df.columns]

        if not available_metrics:
            logger.debug(f"No metrics available for category {category}")
            return None

        # For each metric, calculate the rank
        rank_cols = []
        for metric in available_metrics:
            rank_col = f"{metric} Rank"
            if category == "risk":
                # Lower is better for risk metrics
                ranking_df[rank_col] = self.combined_df[metric].rank(
                    method="min", na_option="keep"
                )
            else:
                # Higher is better for returns and Sharpe
                ranking_df[rank_col] = self.combined_df[metric].rank(
                    ascending=False, method="min", na_option="keep"
                )
            rank_cols.append(rank_col)

        # Calculate average rank for this category
        category_rank = f"{category.capitalize()} Rank"
        if rank_cols:
            ranking_df[category_rank] = ranking_df[rank_cols].mean(axis=1)
            logger.debug(f"Calculated {category_rank} using {len(rank_cols)} metrics")
            return category_rank

        return None

    def generate_visualizations(self) -> None:
        """
        Generate various visualizations for analysis.

        This method creates six key visualizations:
        1. Returns comparison across different time periods
        2. Risk-return scatter plot
        3. Top performing indices
        4. Risk metrics comparison
        5. Sharpe ratio comparison
        6. Correlation heatmap of key metrics

        All visualizations are saved to the output directory as PNG files.
        """
        logger.info("Generating visualizations...")

        # Set visualization style for consistency
        self._set_visualization_style()

        # Generate each visualization, catching errors individually
        visualization_methods = [
            ("returns comparison", self._plot_returns_comparison),
            ("risk-return scatter", self._plot_risk_return_scatter),
            ("top performers", self._plot_top_performers),
            ("risk metrics", self._plot_risk_metrics),
            ("Sharpe ratio comparison", self._plot_sharpe_ratio_comparison),
            ("correlation heatmap", self._plot_correlation_heatmap),
        ]

        for viz_name, viz_method in visualization_methods:
            try:
                viz_method()
                logger.debug(f"Generated {viz_name} visualization")
            except Exception as e:
                logger.error(f"Error generating {viz_name} visualization: {e}")

        logger.info("Completed generating visualizations")

    def _set_visualization_style(self) -> None:
        """Set consistent style for all visualizations."""
        # Use seaborn's whitegrid style for clean, professional plots
        sns.set(style="whitegrid", palette="muted", font_scale=1.2)

        # Default figure size for readability
        plt.rcParams["figure.figsize"] = (14, 8)

        # Set DPI for print-quality images
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 300

        # Improve font appearance
        plt.rcParams["font.family"] = "sans-serif"

        # Ensure plots are readable in both light and dark backgrounds
        plt.rcParams["axes.edgecolor"] = "#333333"
        plt.rcParams["axes.labelcolor"] = "#333333"
        plt.rcParams["xtick.color"] = "#333333"
        plt.rcParams["ytick.color"] = "#333333"

    def _plot_returns_comparison(self):
        """Plot comparison of returns across different timeframes"""
        if self.returns_df is None or self.returns_df.empty:
            return

        plt.figure(figsize=(16, 10))

        # Select columns to plot
        return_cols = [
            "1 Year Total Return",
            "5 Years Total Return",
            "Since Inception Total Return",
        ]
        return_cols = [col for col in return_cols if col in self.returns_df.columns]

        if not return_cols:
            return

        # Get top 15 indices by 5 year return
        sort_col = (
            "5 Years Total Return"
            if "5 Years Total Return" in return_cols
            else return_cols[0]
        )
        top_indices = (
            self.returns_df[return_cols]
            .dropna()
            .sort_values(sort_col, ascending=False)
            .head(15)
        )

        # Plot
        ax = top_indices.plot(kind="bar", width=0.8)
        plt.title("Top 15 Indices by Total Return")
        plt.xlabel("Index")
        plt.ylabel("Return (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.legend(title="Time Period")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Save plot
        plt.savefig(
            self.output_dir / "returns_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_risk_return_scatter(self):
        """Create a risk-return scatter plot"""
        if self.combined_df is None or self.combined_df.empty:
            return

        # Check if we have the necessary columns
        if (
            "5 Years Total Return" not in self.combined_df.columns
            or "Std. Deviation (5 Years)" not in self.combined_df.columns
        ):
            return

        plt.figure(figsize=(12, 10))

        # Create a scatter plot
        data = self.combined_df.dropna(
            subset=["5 Years Total Return", "Std. Deviation (5 Years)"]
        )

        # Get data for plotting
        x = data["Std. Deviation (5 Years)"]
        y = data["5 Years Total Return"]

        # Create scatter plot with different sizes based on Sharpe ratio if available
        if "Sharpe Ratio (5 Years)" in data.columns:
            sizes = data["Sharpe Ratio (5 Years)"] * 50
            sizes = sizes.clip(lower=20, upper=300)  # Limit size range
        else:
            sizes = 100

        # Create the scatter plot
        scatter = plt.scatter(x, y, s=sizes, alpha=0.7, c=y, cmap="viridis")

        # Add labels for some points
        top_indices = data.nlargest(10, "5 Years Total Return").index
        low_risk = data.nsmallest(5, "Std. Deviation (5 Years)").index
        high_sharpe = (
            data.nlargest(5, "Sharpe Ratio (5 Years)").index
            if "Sharpe Ratio (5 Years)" in data.columns
            else []
        )

        # Create a set of all indices to label
        indices_to_label = set(list(top_indices) + list(low_risk) + list(high_sharpe))

        for idx in indices_to_label:
            plt.annotate(
                idx,
                (
                    data.loc[idx, "Std. Deviation (5 Years)"],
                    data.loc[idx, "5 Years Total Return"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

        # Add reference lines
        plt.axhline(
            y=data["5 Years Total Return"].median(),
            color="r",
            linestyle="--",
            alpha=0.3,
        )
        plt.axvline(
            x=data["Std. Deviation (5 Years)"].median(),
            color="r",
            linestyle="--",
            alpha=0.3,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("5 Year Return (%)")

        # Add labels and title
        plt.xlabel("Risk (5-Year Standard Deviation)")
        plt.ylabel("5-Year Total Return (%)")
        plt.title("Risk vs. Return Analysis (5-Year)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(
            self.output_dir / "risk_return_scatter.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_top_performers(self):
        """Plot top performing indices across different time periods"""
        if self.ranking_df is None or self.ranking_df.empty or self.combined_df is None:
            return

        # Get top 10 indices by overall rank
        if "Overall Rank" in self.ranking_df.columns:
            top_indices = self.ranking_df.sort_values("Overall Rank").head(10).index

            plt.figure(figsize=(14, 10))

            # For each top index, plot its returns across time periods
            periods = [
                "1 Year Total Return",
                "5 Years Total Return",
                "Since Inception Total Return",
            ]
            periods = [p for p in periods if p in self.combined_df.columns]

            if not periods:
                return

            # Extract data for plotting
            plot_data = self.combined_df.loc[top_indices, periods].copy()

            # Create bar chart
            ax = plot_data.plot(kind="barh", width=0.7)
            plt.title("Top 10 Overall Indices - Performance Comparison")
            plt.xlabel("Return (%)")
            plt.ylabel("Index")
            plt.grid(axis="x", linestyle="--", alpha=0.7)
            plt.legend(title="Time Period")
            plt.tight_layout()

            # Save plot
            plt.savefig(
                self.output_dir / "top_performers.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def _plot_risk_metrics(self):
        """Plot risk metrics comparison for selected indices"""
        if self.stats_df is None or self.stats_df.empty:
            return

        # Check if we have the necessary columns
        risk_cols = ["Std. Deviation (1 Year)", "Std. Deviation (5 Years)"]
        beta_cols = ["Beta (NIFTY 50) (1 Year)", "Beta (NIFTY 50) (5 Years)"]

        have_risk = all(col in self.stats_df.columns for col in risk_cols)
        have_beta = all(col in self.stats_df.columns for col in beta_cols)

        if not (have_risk or have_beta):
            return

        # Get indices with lowest and highest risk
        if have_risk:
            lowest_risk = self.stats_df.nsmallest(5, "Std. Deviation (5 Years)").index
            highest_risk = self.stats_df.nlargest(5, "Std. Deviation (5 Years)").index
        else:
            lowest_risk = []
            highest_risk = []

        # Get indices with lowest and highest beta
        if have_beta:
            lowest_beta = self.stats_df.nsmallest(5, "Beta (NIFTY 50) (5 Years)").index
            highest_beta = self.stats_df.nlargest(5, "Beta (NIFTY 50) (5 Years)").index
        else:
            lowest_beta = []
            highest_beta = []

        # Combine and get unique indices
        selected_indices = list(
            set(
                list(lowest_risk)
                + list(highest_risk)
                + list(lowest_beta)
                + list(highest_beta)
            )
        )
        selected_indices = selected_indices[:15]  # Limit to 15

        # Get columns to plot
        cols_to_plot = []
        if have_risk:
            cols_to_plot.extend(risk_cols)
        if have_beta:
            cols_to_plot.extend(beta_cols)

        if not selected_indices or not cols_to_plot:
            return

        # Create plot
        plt.figure(figsize=(14, 10))

        # Create grouped bar chart
        plot_data = self.stats_df.loc[selected_indices, cols_to_plot].copy()
        ax = plot_data.plot(kind="bar", width=0.8)

        plt.title("Risk Metrics Comparison - Selected Indices")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Risk Metric")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save plot
        plt.savefig(self.output_dir / "risk_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_sharpe_ratio_comparison(self):
        """Plot comparison of Sharpe ratios"""
        if self.sharpe_ratios is None or self.sharpe_ratios.empty:
            return

        # Get top 15 indices by 5-Year Sharpe ratio
        if "Sharpe Ratio (5 Years)" in self.sharpe_ratios.columns:
            top_indices = self.sharpe_ratios.nlargest(
                15, "Sharpe Ratio (5 Years)"
            ).index

            plt.figure(figsize=(14, 10))

            # Get columns to plot
            cols = [
                "Sharpe Ratio (1 Year)",
                "Sharpe Ratio (5 Years)",
                "Sharpe Ratio (Since Inception)",
            ]
            cols = [c for c in cols if c in self.sharpe_ratios.columns]

            if not cols:
                return

            # Create bar chart
            plot_data = self.sharpe_ratios.loc[top_indices, cols].copy()
            ax = plot_data.plot(kind="bar", width=0.8)

            plt.title("Top 15 Indices by Risk-Adjusted Return (Sharpe Ratio)")
            plt.xlabel("Index")
            plt.ylabel("Sharpe Ratio")
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Time Period")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Save plot
            plt.savefig(
                self.output_dir / "sharpe_ratio_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _plot_correlation_heatmap(self):
        """Plot correlation heatmap of selected metrics"""
        if self.combined_df is None or self.combined_df.empty:
            return

        # Select interesting metrics for correlation analysis
        return_metrics = ["1 Year Total Return", "5 Years Total Return"]
        risk_metrics = ["Std. Deviation (1 Year)", "Std. Deviation (5 Years)"]

        metrics = [
            m for m in return_metrics + risk_metrics if m in self.combined_df.columns
        ]

        if len(metrics) < 2:
            return

        # Calculate correlation matrix
        corr_matrix = self.combined_df[metrics].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
        plt.title("Correlation Matrix of Key Performance Metrics")
        plt.tight_layout()

        # Save plot
        plt.savefig(
            self.output_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def generate_investment_recommendations(self):
        """Generate investment recommendations based on analysis"""
        logger.info("Generating investment recommendations...")

        if self.combined_df is None or self.combined_df.empty:
            logger.warning("No data available for investment recommendations")
            return {}

        recommendations = {"analysis_date": self.analysis_date, "categories": {}}

        # 1. Best Overall Performers
        if "Overall Rank" in self.ranking_df.columns:
            top_overall = self.ranking_df.nsmallest(5, "Overall Rank").index.tolist()
            recommendations["categories"]["best_overall"] = {
                "title": "Best Overall Performers",
                "description": "These indices have the best combination of returns, risk, and risk-adjusted metrics",
                "indices": top_overall,
            }

        # 2. Best Risk-Adjusted Return (Sharpe Ratio)
        if "Sharpe Ratio (5 Years)" in self.sharpe_ratios.columns:
            top_sharpe = self.sharpe_ratios.nlargest(
                5, "Sharpe Ratio (5 Years)"
            ).index.tolist()
            recommendations["categories"]["best_risk_adjusted"] = {
                "title": "Best Risk-Adjusted Returns",
                "description": "These indices provide the highest return per unit of risk taken",
                "indices": top_sharpe,
            }

        # 3. Highest Returns
        if "5 Years Total Return" in self.returns_df.columns:
            top_returns = self.returns_df.nlargest(
                5, "5 Years Total Return"
            ).index.tolist()
            recommendations["categories"]["highest_returns"] = {
                "title": "Highest 5-Year Returns",
                "description": "These indices have delivered the strongest absolute returns over 5 years",
                "indices": top_returns,
            }

        # 4. Lowest Risk
        if "Std. Deviation (5 Years)" in self.stats_df.columns:
            lowest_risk = self.stats_df.nsmallest(
                5, "Std. Deviation (5 Years)"
            ).index.tolist()
            recommendations["categories"]["lowest_risk"] = {
                "title": "Lowest Risk Indices",
                "description": "These indices have shown the least volatility over 5 years",
                "indices": lowest_risk,
            }

        # 5. Defensive Picks (Low Beta)
        if "Beta (NIFTY 50) (5 Years)" in self.stats_df.columns:
            defensive = self.stats_df.nsmallest(
                5, "Beta (NIFTY 50) (5 Years)"
            ).index.tolist()
            recommendations["categories"]["defensive_picks"] = {
                "title": "Defensive Picks (Low Beta)",
                "description": "These indices tend to be less affected by market downturns",
                "indices": defensive,
            }

        # 6. Recovery Plays (Recent Underperformers with Strong Long-Term Returns)
        if all(
            col in self.returns_df.columns
            for col in ["1 Year Total Return", "5 Years Total Return"]
        ):
            # Find indices with negative 1Y but strong 5Y returns
            recovery_candidates = (
                self.returns_df[
                    (self.returns_df["1 Year Total Return"] < 0)
                    & (self.returns_df["5 Years Total Return"] > 15)
                ]
                .nlargest(5, "5 Years Total Return")
                .index.tolist()
            )

            if recovery_candidates:
                recommendations["categories"]["recovery_plays"] = {
                    "title": "Recovery Plays",
                    "description": "Recent underperformers with strong long-term track records",
                    "indices": recovery_candidates,
                }

        # Save recommendations to JSON
        with open(
            self.output_dir / "investment_recommendations.json", "w", encoding="utf-8"
        ) as f:
            json.dump(recommendations, f, indent=2)

        logger.info("Saved investment recommendations")

        return recommendations

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        logger.info("Starting complete analysis pipeline")

        # Load data
        self.load_all_data()

        # Calculate risk-adjusted metrics
        self.calculate_risk_adjusted_metrics()

        # Calculate advanced risk metrics
        self.calculate_advanced_risk_metrics()

        # Create combined dataset
        self.create_combined_dataset()

        # Rank indices
        self.rank_indices()

        # Perform sophisticated analyses
        try:
            # Perform cluster analysis
            self.perform_cluster_analysis()
            logger.info("Cluster analysis completed")
        except Exception as e:
            logger.error(f"Error in cluster analysis: {e}")

        try:
            # Perform factor analysis
            self.perform_factor_analysis()
            logger.info("Factor analysis completed")
        except Exception as e:
            logger.error(f"Error in factor analysis: {e}")

        try:
            # Build optimal portfolios
            self.build_optimal_portfolios()
            logger.info("Portfolio optimization completed")
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")

        # Generate standard visualizations
        self.generate_visualizations()

        # Generate investment recommendations
        self.generate_investment_recommendations()

        logger.info("Analysis pipeline completed successfully")
        logger.info(f"All results saved to {self.output_dir}")

    def perform_cluster_analysis(self):
        """
        Perform cluster analysis to group indices with similar risk-return characteristics
        using K-Means clustering and hierarchical clustering
        """
        logger.info("Performing cluster analysis on indices...")

        if self.combined_df is None or self.combined_df.empty:
            logger.warning("No data available for cluster analysis")
            return {}

        # Select features for clustering
        features = [
            "5 Years Total Return",
            "Std. Deviation (5 Years)",
            "Beta (NIFTY 50) (5 Years)",
            "1 Year Total Return",
        ]

        # Verify we have these columns
        available_features = [f for f in features if f in self.combined_df.columns]

        if len(available_features) < 2:
            logger.warning(
                f"Not enough features for clustering. Need at least 2, found {len(available_features)}"
            )
            return {}

        # Extract data for clustering
        cluster_data = self.combined_df[available_features].copy()

        # Handle missing values - needed for clustering algorithms
        cluster_data = cluster_data.dropna()

        if len(cluster_data) < 10:
            logger.warning(
                f"Not enough complete data points for clustering. Need at least 10, found {len(cluster_data)}"
            )
            return {}

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        # K-Means Clustering
        kmeans_results = self._perform_kmeans_clustering(scaled_data, cluster_data)

        # Hierarchical Clustering
        hierarchical_results = self._perform_hierarchical_clustering(
            scaled_data, cluster_data
        )

        # Store results in a dictionary
        clustering_results = {
            "kmeans": kmeans_results,
            "hierarchical": hierarchical_results,
            "feature_importance": self._calculate_feature_importance(cluster_data),
        }

        # Save clustering results
        with open(
            self.output_dir / "cluster_analysis.json", "w", encoding="utf-8"
        ) as f:
            # Convert any non-serializable objects to strings or lists
            json_compatible = {}
            for method, results in clustering_results.items():
                json_compatible[method] = {}
                for key, value in results.items():
                    # If value is a numpy array or pandas Series/DataFrame, convert to list
                    if isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)):
                        json_compatible[method][key] = (
                            value.tolist() if hasattr(value, "tolist") else list(value)
                        )
                    else:
                        json_compatible[method][key] = value
            json.dump(json_compatible, f, indent=2)

        # Create the cluster visualization
        self._visualize_clusters(
            scaled_data,
            kmeans_results["labels"],
            cluster_data.index,
            "K-Means Clustering of Indices",
        )

        logger.info("Cluster analysis completed")
        return clustering_results

    def _perform_kmeans_clustering(self, scaled_data, original_data):
        """
        Perform K-Means clustering on the indices data

        Args:
            scaled_data: Standardized data for clustering
            original_data: Original data with index names

        Returns:
            Dictionary with clustering results
        """
        # Determine optimal number of clusters using silhouette scores
        sil_scores = []
        max_clusters = min(
            10, len(scaled_data) - 1
        )  # Can't have more clusters than data points

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_data)

            # Calculate silhouette score
            if len(set(labels)) > 1:  # Ensure we have at least 2 clusters
                sil_score = silhouette_score(scaled_data, labels)
                sil_scores.append((n_clusters, sil_score))

        # Find best number of clusters
        best_n_clusters = max(sil_scores, key=lambda x: x[1])[0] if sil_scores else 2

        # Train final model with optimal clusters
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)

        # Get cluster centers in original scale
        centers_scaled = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers_scaled)

        # Create cluster profiles
        cluster_profiles = {}
        for i in range(best_n_clusters):
            # Indices in this cluster
            indices_in_cluster = original_data.index[labels == i].tolist()

            # Average values for this cluster
            cluster_data = original_data.loc[indices_in_cluster].mean()

            # Profile description
            if (
                "5 Years Total Return" in original_data.columns
                and "Std. Deviation (5 Years)" in original_data.columns
            ):
                avg_return = cluster_data["5 Years Total Return"]
                avg_risk = cluster_data["Std. Deviation (5 Years)"]

                if avg_return > original_data["5 Years Total Return"].mean():
                    return_profile = "high-return"
                else:
                    return_profile = "low-return"

                if avg_risk > original_data["Std. Deviation (5 Years)"].mean():
                    risk_profile = "high-risk"
                else:
                    risk_profile = "low-risk"

                profile = f"{return_profile}-{risk_profile}"
            else:
                profile = f"Cluster {i+1}"

            # Store in profiles
            cluster_profiles[i] = {
                "indices": indices_in_cluster,
                "average_metrics": cluster_data.to_dict(),
                "profile": profile,
                "size": len(indices_in_cluster),
            }

        return {
            "optimal_clusters": best_n_clusters,
            "labels": labels,
            "centers": centers_original,
            "profiles": cluster_profiles,
            "silhouette_scores": dict(sil_scores),
        }

    def _perform_hierarchical_clustering(self, scaled_data, original_data):
        """
        Perform hierarchical clustering on indices data

        Args:
            scaled_data: Standardized data for clustering
            original_data: Original data with index names

        Returns:
            Dictionary with clustering results
        """
        # Compute the distance matrix
        dist_matrix = pdist(scaled_data, metric="euclidean")

        # Perform hierarchical clustering
        linkage_matrix = sch.linkage(dist_matrix, method="ward")

        # Determine optimal number of clusters
        max_d = 0.5 * np.max(linkage_matrix[:, 2])  # 50% of max distance as threshold
        clusters = sch.fcluster(linkage_matrix, max_d, criterion="distance")
        n_clusters = len(set(clusters))

        # Create dendogram visualization
        plt.figure(figsize=(16, 10))
        plt.title("Hierarchical Clustering Dendrogram", fontsize=16)
        plt.xlabel("Index", fontsize=14)
        plt.ylabel("Distance", fontsize=14)

        # Make a more detailed dendrogram with labels
        if len(original_data) <= 50:  # Only show leaf labels if not too many
            sch.dendrogram(
                linkage_matrix,
                leaf_rotation=90,
                leaf_font_size=10,
                labels=original_data.index,
            )
        else:
            sch.dendrogram(linkage_matrix)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "hierarchical_clustering_dendrogram.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create cluster profiles
        cluster_profiles = {}
        for i in range(1, n_clusters + 1):
            # Indices in this cluster
            indices_in_cluster = original_data.index[clusters == i].tolist()

            # Average values for this cluster
            cluster_data = original_data.loc[indices_in_cluster].mean()

            # Store in profiles
            cluster_profiles[i - 1] = {
                "indices": indices_in_cluster,
                "average_metrics": cluster_data.to_dict(),
                "size": len(indices_in_cluster),
            }

        return {
            "n_clusters": n_clusters,
            "labels": clusters,
            "profiles": cluster_profiles,
            "linkage_matrix": linkage_matrix.tolist(),
        }

    def _calculate_feature_importance(self, data):
        """Calculate feature importance for clustering"""
        # Use PCA to determine feature importance
        pca = PCA()
        pca.fit(StandardScaler().fit_transform(data))

        # Get the loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
            index=data.columns,
        )

        # Calculate variance explained
        explained_variance = pd.Series(
            pca.explained_variance_ratio_,
            index=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        )

        # Feature importance based on first principal component
        feature_importance = abs(loadings["PC1"])
        feature_importance = feature_importance / feature_importance.sum()

        return {
            "feature_importance": feature_importance.to_dict(),
            "explained_variance": explained_variance.to_dict(),
            "loadings": loadings.to_dict(),
        }

    def _visualize_clusters(self, scaled_data, labels, index_names, title):
        """
        Visualize clusters using dimensionality reduction

        Args:
            scaled_data: Standardized data used for clustering
            labels: Cluster labels
            index_names: Names of indices for labeling
            title: Plot title
        """
        # Use PCA to reduce dimensions for visualization
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)

        # Create a dataframe for plotting
        plot_df = pd.DataFrame(
            {
                "PC1": reduced_data[:, 0],
                "PC2": reduced_data[:, 1],
                "Cluster": labels,
                "Index": index_names,
            }
        )

        # Create plot
        plt.figure(figsize=(12, 10))

        # Create scatter plot with different colors for each cluster
        sns.scatterplot(
            x="PC1",
            y="PC2",
            hue="Cluster",
            palette="viridis",
            data=plot_df,
            s=100,
            alpha=0.7,
        )

        # Add labels for indices (limit to avoid clutter)
        if len(plot_df) <= 30:  # Only label if not too many points
            for i, row in plot_df.iterrows():
                plt.annotate(
                    row["Index"],
                    (row["PC1"], row["PC2"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                )

        # Add title and axis labels
        plt.title(title, fontsize=16)
        plt.xlabel(
            f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)",
            fontsize=12,
        )
        plt.ylabel(
            f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)",
            fontsize=12,
        )
        plt.tight_layout()

        # Save plot
        plt.savefig(
            self.output_dir / "cluster_analysis_pca.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def calculate_advanced_risk_metrics(self):
        """Calculate advanced risk and performance metrics"""
        logger.info("Calculating advanced risk metrics...")

        if self.combined_df is None or self.combined_df.empty:
            logger.warning("No data available for advanced risk metrics")
            return

        # Create a dataframe for advanced metrics
        advanced_metrics = pd.DataFrame(index=self.combined_df.index)

        # 1. Calculate Sortino Ratio (if we have downside deviation)
        # Sortino is like Sharpe but only penalizes downside volatility
        if "1 Year Total Return" in self.combined_df.columns:
            # We'll assume a minimal acceptable return (MAR) of 3% annually
            mar = 3.0
            returns = self.combined_df["1 Year Total Return"]

            # Estimate downside deviation (simplified calculation)
            # True calculation would use all monthly returns below MAR
            if "Std. Deviation (1 Year)" in self.combined_df.columns:
                # Estimate downside deviation as 60% of total deviation for negative returns
                downside_mask = returns < mar

                if downside_mask.sum() > 0:
                    # For indices with returns below MAR, calculate sortino
                    advanced_metrics["Sortino Ratio (1 Year)"] = np.nan
                    std_dev = self.combined_df.loc[
                        downside_mask, "Std. Deviation (1 Year)"
                    ]

                    # Approximate downside deviation (this is an approximation)
                    downside_dev = std_dev * 0.6

                    # Calculate Sortino ratio
                    advanced_metrics.loc[downside_mask, "Sortino Ratio (1 Year)"] = (
                        returns[downside_mask] - mar
                    ) / downside_dev

        # 2. Calculate Maximum Drawdown (if possible)
        # Without full price history, estimate based on annual returns volatility
        if "Std. Deviation (5 Years)" in self.combined_df.columns:
            # Approximate max drawdown as 2x the annual std deviation
            # This is a statistical approximation assuming normal distribution
            advanced_metrics["Estimated Max Drawdown (5 Years)"] = (
                self.combined_df["Std. Deviation (5 Years)"] * 2.0
            )

        # 3. Calculate Information Ratio (active return / tracking error)
        # Relative to NIFTY 50
        if all(
            col in self.combined_df.columns
            for col in [
                "1 Year Total Return",
                "Beta (NIFTY 50) (1 Year)",
                "Correlation (NIFTY 50) (1 Year)",
            ]
        ):

            # This is an approximation; true calculation needs return history
            # Assuming benchmark (NIFTY 50) return of 8% for the year
            benchmark_return = 8.0

            # Calculate active return
            active_return = self.combined_df["1 Year Total Return"] - benchmark_return

            # Estimate tracking error using beta and correlation
            beta = self.combined_df["Beta (NIFTY 50) (1 Year)"]
            correlation = self.combined_df["Correlation (NIFTY 50) (1 Year)"]

            # Tracking error approximation using the relationship between beta and correlation
            # True tracking error requires return time series
            if "Std. Deviation (1 Year)" in self.combined_df.columns:
                std_dev = self.combined_df["Std. Deviation (1 Year)"]
                tracking_error = std_dev * np.sqrt(1 - correlation**2)

                # Calculate information ratio
                advanced_metrics["Information Ratio (1 Year)"] = (
                    active_return / tracking_error
                )

        # 4. Calculate Treynor Ratio (return per unit of systematic risk)
        if all(
            col in self.combined_df.columns
            for col in ["1 Year Total Return", "Beta (NIFTY 50) (1 Year)"]
        ):

            # Risk free rate assumption
            risk_free = 3.0

            # Calculate excess return
            excess_return = self.combined_df["1 Year Total Return"] - risk_free

            # Calculate Treynor ratio
            beta = self.combined_df["Beta (NIFTY 50) (1 Year)"]
            advanced_metrics["Treynor Ratio (1 Year)"] = excess_return / beta

        # 5. Tail Risk Estimate (Value at Risk approximation)
        if "Std. Deviation (1 Year)" in self.combined_df.columns:
            # 95% VaR approximation assuming normal distribution
            # VaR = μ - z*σ where z is the z-score for the confidence level
            z_score_95 = 1.645

            if "1 Year Total Return" in self.combined_df.columns:
                mean_return = self.combined_df["1 Year Total Return"]
                std_dev = self.combined_df["Std. Deviation (1 Year)"]

                # Calculate 95% VaR (simplified)
                advanced_metrics["95% VaR (1 Year)"] = -(
                    mean_return - z_score_95 * std_dev
                )

        # 6. Calmar Ratio (if we have drawdown estimates)
        if (
            "Estimated Max Drawdown (5 Years)" in advanced_metrics.columns
            and "5 Years Total Return" in self.combined_df.columns
        ):
            # Calculate annualized return over 5 years
            cagr_5yr = (
                (1 + self.combined_df["5 Years Total Return"] / 100) ** (1 / 5)
            ) - 1
            cagr_5yr = cagr_5yr * 100  # convert to percentage

            # Calculate Calmar ratio
            advanced_metrics["Calmar Ratio"] = (
                cagr_5yr / advanced_metrics["Estimated Max Drawdown (5 Years)"]
            )

        # Merge advanced metrics with combined data
        self.advanced_metrics = advanced_metrics

        # Add to combined dataset if it exists
        if self.combined_df is not None:
            for col in advanced_metrics.columns:
                self.combined_df[col] = advanced_metrics[col]

        logger.info(f"Calculated {len(advanced_metrics.columns)} advanced risk metrics")

    def perform_factor_analysis(self):
        """
        Perform factor analysis to identify underlying factors driving index returns
        """
        logger.info("Performing factor analysis...")

        if self.combined_df is None or self.combined_df.empty:
            logger.warning("No data available for factor analysis")
            return

        # Select features for factor analysis
        potential_features = [
            "1 Year Total Return",
            "5 Years Total Return",
            "Std. Deviation (1 Year)",
            "Std. Deviation (5 Years)",
            "Beta (NIFTY 50) (1 Year)",
            "Beta (NIFTY 50) (5 Years)",
            "Correlation (NIFTY 50) (1 Year)",
            "Correlation (NIFTY 50) (5 Years)",
        ]

        # Check which features are available
        features = [f for f in potential_features if f in self.combined_df.columns]

        if len(features) < 3:
            logger.warning(
                f"Not enough features for factor analysis. Need at least 3, found {len(features)}"
            )
            return

        # Extract clean data for factor analysis
        factor_data = self.combined_df[features].copy().dropna()

        if len(factor_data) < 10:
            logger.warning(
                f"Not enough complete data points for factor analysis. Need at least 10, found {len(factor_data)}"
            )
            return

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(factor_data)

        # Determine optimal number of components
        pca = PCA()
        pca.fit(scaled_data)

        # Decide number of factors using explained variance threshold (e.g., 80%)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        n_components = np.argmax(cumulative_variance >= 0.8) + 1

        # Make sure we have at least 2 components
        n_components = max(2, min(n_components, len(features) - 1))

        # Run PCA with optimal components
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)

        # Create dataframe with principal components
        pc_df = pd.DataFrame(
            principal_components,
            index=factor_data.index,
            columns=[f"Factor_{i+1}" for i in range(n_components)],
        )

        # Create loadings dataframe
        loadings = pd.DataFrame(
            pca.components_.T,
            index=features,
            columns=[f"Factor_{i+1}" for i in range(n_components)],
        )

        # Determine factor names based on loadings
        factor_names = {}
        for i in range(n_components):
            factor_col = f"Factor_{i+1}"

            # Get top features with high positive and negative loadings
            top_pos = loadings[factor_col].nlargest(2).index.tolist()
            top_neg = loadings[factor_col].nsmallest(2).index.tolist()

            # Create factor name based on these features
            if "Return" in "".join(top_pos) and "Std. Deviation" in "".join(
                top_pos + top_neg
            ):
                factor_names[factor_col] = "Risk-Return Factor"
            elif "Beta" in "".join(top_pos + top_neg):
                factor_names[factor_col] = "Market Sensitivity Factor"
            elif "Correlation" in "".join(top_pos + top_neg):
                factor_names[factor_col] = "Market Correlation Factor"
            elif "Return" in "".join(top_pos + top_neg):
                factor_names[factor_col] = "Return Factor"
            else:
                factor_names[factor_col] = f"Factor {i+1}"

        # Visualize the loadings
        plt.figure(figsize=(12, 8))
        sns.heatmap(loadings, cmap="coolwarm", annot=True, fmt=".2f", center=0)
        plt.title("Factor Analysis: Feature Loadings", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "factor_analysis_loadings.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Visualize the principal components
        if n_components >= 2:
            plt.figure(figsize=(10, 8))

            # Get top indices by Factor 1 and Factor 2 (positive and negative)
            top_f1_pos = pc_df.nlargest(5, "Factor_1").index
            top_f1_neg = pc_df.nsmallest(5, "Factor_1").index
            top_f2_pos = pc_df.nlargest(5, "Factor_2").index
            top_f2_neg = pc_df.nsmallest(5, "Factor_2").index

            # Combine all for labeling
            indices_to_label = set(
                list(top_f1_pos)
                + list(top_f1_neg)
                + list(top_f2_pos)
                + list(top_f2_neg)
            )

            # Create scatter plot
            sns.scatterplot(x="Factor_1", y="Factor_2", data=pc_df, s=50)

            # Add labels
            for idx in indices_to_label:
                plt.annotate(
                    idx,
                    (pc_df.loc[idx, "Factor_1"], pc_df.loc[idx, "Factor_2"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                )

            # Update axis labels with factor names
            plt.xlabel(
                f"{factor_names.get('Factor_1', 'Factor 1')} ({explained_variance[0]:.1%})",
                fontsize=12,
            )
            plt.ylabel(
                f"{factor_names.get('Factor_2', 'Factor 2')} ({explained_variance[1]:.1%})",
                fontsize=12,
            )
            plt.title("Factor Analysis of Index Characteristics", fontsize=16)
            plt.tight_layout()
            plt.savefig(
                self.output_dir / "factor_analysis_scatter.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # Save factor analysis results
        factor_analysis_results = {
            "explained_variance": explained_variance.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
            "n_components": n_components,
            "loadings": loadings.to_dict(),
            "factor_names": factor_names,
            "principal_components": pc_df.to_dict(),
            "features_used": features,
        }

        with open(self.output_dir / "factor_analysis.json", "w", encoding="utf-8") as f:
            json.dump(factor_analysis_results, f, indent=2)

        logger.info(f"Factor analysis completed with {n_components} factors identified")
        return factor_analysis_results

    def build_optimal_portfolios(self):
        """
        Build optimal portfolios based on different investment objectives
        - Maximum Sharpe ratio portfolio (efficient portfolio)
        - Minimum volatility portfolio
        - Maximum return portfolio
        - Various risk-targeted portfolios
        """
        logger.info("Building optimal portfolios...")

        if self.combined_df is None or self.combined_df.empty:
            logger.warning("No data available for portfolio optimization")
            return

        # Check if we have necessary return and risk metrics
        required_cols = ["5 Years Total Return", "Std. Deviation (5 Years)"]
        if not all(col in self.combined_df.columns for col in required_cols):
            logger.warning(
                "Missing required return or risk metrics for portfolio optimization"
            )
            return

        # Extract clean data for optimization
        portfolio_data = self.combined_df[required_cols].copy().dropna()

        if len(portfolio_data) < 5:
            logger.warning(
                f"Not enough complete data points for portfolio optimization. Need at least 5, found {len(portfolio_data)}"
            )
            return

        # Annualize the 5-year returns
        returns = (
            portfolio_data["5 Years Total Return"] / 100
        )  # Convert percentage to decimal
        annual_returns = (
            (1 + returns) ** (1 / 5)
        ) - 1  # Convert 5-year to annualized return

        # Get volatility (already annualized)
        volatility = (
            portfolio_data["Std. Deviation (5 Years)"] / 100
        )  # Convert to decimal

        # Create dataframe for portfolio optimization
        opt_data = pd.DataFrame(
            {"Annual Return": annual_returns, "Annual Volatility": volatility}
        )

        # 1. Maximum Sharpe Ratio Portfolio (assuming risk-free rate of 3%)
        risk_free_rate = 0.03
        sharpe_ratios = (opt_data["Annual Return"] - risk_free_rate) / opt_data[
            "Annual Volatility"
        ]
        max_sharpe_idx = sharpe_ratios.idxmax()
        max_sharpe_indices = [max_sharpe_idx]

        # 2. Minimum Volatility Portfolio (single index with lowest volatility)
        min_vol_idx = opt_data["Annual Volatility"].idxmin()
        min_vol_indices = [min_vol_idx]

        # 3. Maximum Return Portfolio (single index with highest return)
        max_return_idx = opt_data["Annual Return"].idxmax()
        max_return_indices = [max_return_idx]

        # 4. Build Balanced Portfolio (multiple indices)
        # Since we don't have correlation data between all indices, we'll use a heuristic approach
        # Select indices with good Sharpe ratios but from different clusters

        # If we have clustering results, use them to select diverse indices
        diverse_portfolio = []

        # Try to create a portfolio with 4-5 diverse indices with good Sharpe ratios
        top_sharpe = sharpe_ratios.nlargest(15).index

        # Select top indices from different sectors/clusters if possible
        # This is simplified - real portfolio construction would use modern portfolio theory
        # and optimization with correlation matrices

        # If we don't have cluster analysis, just pick top 5 by Sharpe ratio
        diverse_portfolio = top_sharpe[:5].tolist()

        # Store all portfolios
        portfolios = {
            "max_sharpe": {
                "indices": max_sharpe_indices,
                "weights": [1.0] * len(max_sharpe_indices),
                "expected_return": float(opt_data.loc[max_sharpe_idx, "Annual Return"]),
                "expected_volatility": float(
                    opt_data.loc[max_sharpe_idx, "Annual Volatility"]
                ),
                "sharpe_ratio": float(sharpe_ratios.loc[max_sharpe_idx]),
            },
            "min_volatility": {
                "indices": min_vol_indices,
                "weights": [1.0] * len(min_vol_indices),
                "expected_return": float(opt_data.loc[min_vol_idx, "Annual Return"]),
                "expected_volatility": float(
                    opt_data.loc[min_vol_idx, "Annual Volatility"]
                ),
                "sharpe_ratio": float(sharpe_ratios.loc[min_vol_idx]),
            },
            "max_return": {
                "indices": max_return_indices,
                "weights": [1.0] * len(max_return_indices),
                "expected_return": float(opt_data.loc[max_return_idx, "Annual Return"]),
                "expected_volatility": float(
                    opt_data.loc[max_return_idx, "Annual Volatility"]
                ),
                "sharpe_ratio": float(sharpe_ratios.loc[max_return_idx]),
            },
            "balanced": {
                "indices": diverse_portfolio,
                "weights": [1 / len(diverse_portfolio)] * len(diverse_portfolio),
                "expected_return": float(
                    opt_data.loc[diverse_portfolio, "Annual Return"].mean()
                ),
                "expected_volatility": float(
                    opt_data.loc[diverse_portfolio, "Annual Volatility"].mean()
                ),
                # This is a simplified approximation; true portfolio vol would need correlation matrix
                "description": "Diversified portfolio of indices with strong risk-adjusted returns",
            },
        }

        # Visualize the efficient frontier (simplified)
        plt.figure(figsize=(10, 8))
        plt.scatter(
            opt_data["Annual Volatility"] * 100,  # Back to percentage
            opt_data["Annual Return"] * 100,  # Back to percentage
            s=50,
            alpha=0.6,
        )

        # Mark the selected portfolios
        portfolios_to_mark = [
            ("max_sharpe", "Max Sharpe", "red", "^"),
            ("min_volatility", "Min Volatility", "green", "s"),
            ("max_return", "Max Return", "orange", "*"),
        ]

        for port_id, label, color, marker in portfolios_to_mark:
            plt.scatter(
                portfolios[port_id]["expected_volatility"] * 100,
                portfolios[port_id]["expected_return"] * 100,
                s=200,
                color=color,
                marker=marker,
                label=label,
            )

        # Add labels for the indices in each portfolio
        for port_id, port_data in portfolios.items():
            # Skip balanced portfolio as it might have too many indices
            if port_id != "balanced":
                for idx in port_data["indices"]:
                    vol = opt_data.loc[idx, "Annual Volatility"] * 100
                    ret = opt_data.loc[idx, "Annual Return"] * 100

                    plt.annotate(
                        idx,
                        (vol, ret),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                    )

        plt.title("Portfolio Optimization and Efficient Frontier", fontsize=16)
        plt.xlabel("Annual Volatility (%)", fontsize=12)
        plt.ylabel("Expected Annual Return (%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            self.output_dir / "portfolio_optimization.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Save portfolios to JSON
        with open(
            self.output_dir / "optimal_portfolios.json", "w", encoding="utf-8"
        ) as f:
            json.dump(portfolios, f, indent=2)

        logger.info("Portfolio optimization completed")
        return portfolios

    def run_complete_analysis(self) -> bool:
        """
        Run the complete analysis pipeline.

        This is the main method to call for executing the entire analysis process.
        It executes each step in sequence, handling errors at each stage.

        Returns:
            bool: True if analysis completed successfully, False otherwise
        """
        logger.info("Starting complete analysis pipeline")
        success = True

        try:
            # Step 1: Load data
            self.load_all_data()

            # Check if we have data to analyze
            if self.returns_df is None or self.returns_df.empty:
                logger.error("No return data available for analysis")
                return False

            # Step 2: Calculate risk-adjusted metrics
            self.calculate_risk_adjusted_metrics()

            # Step 3: Create combined dataset
            self.create_combined_dataset()

            # Step 4: Rank indices
            self.rank_indices()

            # Step 5: Generate visualizations
            self.generate_visualizations()

            # Step 6: Generate recommendations
            recommendations = self.generate_investment_recommendations()
            if not recommendations:
                logger.warning("No recommendations were generated")

            logger.info("Analysis pipeline completed successfully")
            logger.info(f"Results saved to {self.output_dir}")

            return True

        except Exception as e:
            logger.exception(f"Error in analysis pipeline: {e}")
            return False


def main() -> int:
    """
    Main function for command-line execution.

    This function:
    1. Parses command-line arguments
    2. Configures logging
    3. Initializes the analyzer
    4. Runs the complete analysis
    5. Reports execution time

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(
            description="Analyze NSE indices from factsheet data",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--json-dir",
            default="parsed_factsheets",
            help="Directory containing parsed factsheet JSON files",
        )
        parser.add_argument(
            "--output-dir",
            default="analysis_results",
            help="Directory to save analysis results and visualizations",
        )
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Set the logging level",
        )
        parser.add_argument(
            "--risk-free-rate",
            type=float,
            default=3.0,
            help="Risk-free rate for Sharpe ratio calculation (percentage)",
        )

        args = parser.parse_args()

        # Check if json-dir exists
        json_dir_path = Path(args.json_dir)
        if not json_dir_path.exists():
            logger.error(f"JSON directory not found: {json_dir_path}")
            return 1

        # Set log level from command line
        logger.setLevel(getattr(logging, args.log_level))

        # Start time measurement
        start_time = datetime.now()

        # Run analysis
        logger.info(f"Starting NSE Indices Analysis at {start_time}")
        analyzer = IndicesAnalyzer(json_dir=args.json_dir, output_dir=args.output_dir)
        success = analyzer.run_complete_analysis()

        # End time and duration
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(
            f"Analysis {'completed successfully' if success else 'failed'} in {duration}"
        )

        # Report output location if successful
        if success:
            logger.info(f"Results available in {Path(args.output_dir).absolute()}")

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
