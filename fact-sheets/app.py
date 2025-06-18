#!/usr/bin/env python
"""
NSE Indices Analysis Flask Web Application

This script provides a web interface for analyzing and visualizing performance metrics
across all NSE index factsheets using Flask framework.
"""

import os
import json
import shutil
from pathlib import Path
import logging
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    redirect,
    url_for,
)
from analyze_indices import IndicesAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["TEMPLATES_AUTO_RELOAD"] = True


# Add context processor to provide common variables to all templates
@app.context_processor
def inject_now():
    return {"now": datetime.now()}


# Directory paths
JSON_DIR = Path("parsed_factsheets")
OUTPUT_DIR = Path("analysis_results")
STATIC_DIR = Path("static")

# Create required directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)


def get_analyzer():
    """Initialize and return an IndicesAnalyzer instance"""
    analyzer = IndicesAnalyzer(json_dir=JSON_DIR, output_dir=OUTPUT_DIR)
    return analyzer


def ensure_analysis_results():
    """
    Check if analysis results exist, and if not, run the analysis.
    Returns the analysis date as a string.
    """
    recommendations_file = OUTPUT_DIR / "investment_recommendations.json"
    if not recommendations_file.exists() or not list(STATIC_DIR.glob("*.png")):
        logger.info("Analysis results not found. Running analysis...")
        try:
            analyzer = get_analyzer()
            success = analyzer.run_complete_analysis()

            if success:
                # Copy generated images to static folder for web display
                for img_file in OUTPUT_DIR.glob("*.png"):
                    shutil.copy(img_file, STATIC_DIR / img_file.name)
                analysis_date = analyzer.analysis_date
            else:
                logger.error("Failed to run analysis automatically")
                analysis_date = datetime.now().strftime("%Y-%m-%d")
        except Exception as e:
            logger.exception(f"Error running automatic analysis: {e}")
            analysis_date = datetime.now().strftime("%Y-%m-%d")
    else:
        # Get analysis date from recommendations file if it exists
        try:
            with open(recommendations_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                analysis_date = data.get(
                    "analysis_date", datetime.now().strftime("%Y-%m-%d")
                )
        except Exception:
            analysis_date = datetime.now().strftime("%Y-%m-%d")

    return analysis_date


# Run analysis on startup if needed
analysis_date = ensure_analysis_results()


@app.route("/")
def index():
    """Render the main dashboard page"""
    # Check if we have analysis results
    recommendations_file = OUTPUT_DIR / "investment_recommendations.json"
    has_results = recommendations_file.exists()

    # Get list of factsheets for display
    factsheets = []
    if JSON_DIR.exists():
        factsheets = [f.stem for f in JSON_DIR.glob("*.json")]

    return render_template(
        "index.html",
        has_results=has_results,
        factsheets=factsheets,
        analysis_date=analysis_date,
    )


@app.route("/run_analysis", methods=["POST"])
def run_analysis():
    """Run the complete analysis process"""
    try:
        # Get parameters from form if provided
        risk_free_rate = float(request.form.get("risk_free_rate", "3.0"))

        # Initialize analyzer
        analyzer = get_analyzer()

        # Run the complete analysis pipeline
        success = analyzer.run_complete_analysis()

        if not success:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Analysis failed. Check logs for details.",
                    }
                ),
                500,
            )

        # Copy generated images to static folder for web display
        for img_file in OUTPUT_DIR.glob("*.png"):
            shutil.copy(img_file, STATIC_DIR / img_file.name)

        return jsonify(
            {
                "success": True,
                "message": "Analysis completed successfully",
                "analysis_date": analyzer.analysis_date,
            }
        )

    except Exception as e:
        logger.exception(f"Error running analysis: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500


@app.route("/results")
def results():
    """Display analysis results"""
    # Load recommendations if available
    recommendations = {}
    recommendations_file = OUTPUT_DIR / "investment_recommendations.json"
    if recommendations_file.exists():
        with open(recommendations_file, "r", encoding="utf-8") as f:
            recommendations = json.load(f)

    # Get list of generated charts
    charts = [f.name for f in STATIC_DIR.glob("*.png")]

    # Check if we have a combined dataframe to display as a table
    combined_data = None
    combined_file = OUTPUT_DIR / "combined_data.json"
    if combined_file.exists():
        with open(combined_file, "r", encoding="utf-8") as f:
            combined_data = json.load(f)

    return render_template(
        "results.html",
        recommendations=recommendations,
        charts=charts,
        combined_data=combined_data,
        analysis_date=analysis_date,
    )


@app.route("/api/data")
def api_data():
    """API endpoint to get analysis data in JSON format"""
    data_type = request.args.get("type", "recommendations")

    try:
        if data_type == "recommendations":
            file_path = OUTPUT_DIR / "investment_recommendations.json"
        elif data_type == "combined":
            file_path = OUTPUT_DIR / "combined_data.json"
        else:
            return jsonify({"error": "Invalid data type"}), 400

        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        else:
            return jsonify({"error": "Data not found"}), 404

    except Exception as e:
        logger.error(f"Error fetching API data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files"""
    return send_from_directory(STATIC_DIR, filename)


@app.route("/factsheets")
def factsheets():
    """Display list of available factsheets"""
    if JSON_DIR.exists():
        factsheets = [
            {"name": f.stem, "size": f.stat().st_size} for f in JSON_DIR.glob("*.json")
        ]
    else:
        factsheets = []

    return render_template(
        "factsheets.html", factsheets=factsheets, analysis_date=analysis_date
    )


@app.route("/about")
def about():
    """Display information about the application"""
    return render_template("about.html", analysis_date=analysis_date)


# Save combined data to JSON for table display
def save_combined_data(analyzer):
    """Save the combined dataset to JSON for web display"""
    if analyzer.combined_df is not None and not analyzer.combined_df.empty:
        # Convert to JSON and save
        data = analyzer.combined_df.reset_index().to_dict(orient="records")
        with open(OUTPUT_DIR / "combined_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# Enable analyzer to save the combined dataframe to JSON
def extend_analyzer():
    """Extend IndicesAnalyzer with a method to save combined data"""
    original_create_combined = IndicesAnalyzer.create_combined_dataset

    def extended_create_combined(self):
        original_create_combined(self)
        save_combined_data(self)

    IndicesAnalyzer.create_combined_dataset = extended_create_combined


# Extend the analyzer
extend_analyzer()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
