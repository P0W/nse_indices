#!/usr/bin/env python3
"""
ETF Momentum Strategy Web Server
Elegant web server for ETF portfolio management with file upload support and beautiful UI
"""

import json
import os
import tempfile
import traceback
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import logging

# Configure logging to prevent file creation in web server mode
logging.basicConfig(
    level=logging.WARNING,  # Reduce log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # Only log to stdout, no files
)

# Disable file logging from core module
os.environ["DISABLE_FILE_LOGGING"] = "1"

import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import CLI functions
from cli import (
    show_current_portfolio,
    show_rebalancing_needs,
    show_historical_portfolio,
)


# Pydantic models
class PortfolioRequest(BaseModel):
    amount: float = Field(default=1000000, description="Investment amount in INR")
    size: int = Field(default=5, description="Portfolio size (number of ETFs)")


class HistoricalRequest(BaseModel):
    from_date: str = Field(description="Start date in YYYY-MM-DD format")
    to_date: Optional[str] = Field(
        default=None, description="End date in YYYY-MM-DD format"
    )
    amount: float = Field(default=1000000, description="Investment amount in INR")
    size: int = Field(default=5, description="Portfolio size")


class HoldingItem(BaseModel):
    symbol: str
    units: float
    price: float = Field(
        default=-1, description="Purchase price (-1 to fetch from historical data)"
    )


class RebalanceRequest(BaseModel):
    holdings: list[HoldingItem]
    from_date: Optional[str] = Field(
        default=None, description="Purchase date for price lookup"
    )
    size: int = Field(default=5, description="Portfolio size")


# FastAPI app
app = FastAPI(
    title="ETF Momentum Strategy API",
    description="HTTP API for ETF Momentum Portfolio Management with Beautiful Web UI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory and mount
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def capture_output(func, *args, **kwargs):
    """Capture stdout output from CLI functions"""
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        result = func(*args, **kwargs)
        output = captured_output.getvalue()
        return {"status": "success", "output": output, "result": result}
    except Exception as e:
        output = captured_output.getvalue()
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        return {
            "status": "error",
            "output": output,
            "error": error_msg,
            "traceback": traceback_str,
        }
    finally:
        sys.stdout = old_stdout


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ETF Momentum Strategy API",
        "version": "1.0.0",
        "web_app": "/app",
        "endpoints": {
            "portfolio": "/portfolio",
            "rebalance": "/rebalance",
            "rebalance_upload": "/rebalance/upload",
            "historical": "/historical",
            "health": "/health",
            "docs": "/docs",
            "web_app": "/app",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/portfolio")
async def get_portfolio_get(
    amount: float = Query(default=1000000, description="Investment amount"),
    size: int = Query(default=5, description="Portfolio size"),
):
    """Get current optimal portfolio (GET method)"""
    try:
        result = capture_output(
            show_current_portfolio, investment_amount=amount, portfolio_size=size
        )

        if result["status"] == "success":
            return JSONResponse(
                content={
                    "status": "success",
                    "data": {
                        "amount": amount,
                        "size": size,
                        "output": result["output"],
                    },
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio")
async def get_portfolio_post(request: PortfolioRequest):
    """Get current optimal portfolio (POST method)"""
    try:
        result = capture_output(
            show_current_portfolio,
            investment_amount=request.amount,
            portfolio_size=request.size,
        )

        if result["status"] == "success":
            return JSONResponse(
                content={
                    "status": "success",
                    "data": {
                        "amount": request.amount,
                        "size": request.size,
                        "output": result["output"],
                    },
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebalance")
async def rebalance_portfolio(request: RebalanceRequest):
    """Analyze portfolio rebalancing needs with JSON data"""
    try:
        # Create temporary file for holdings
        holdings_data = []
        for holding in request.holdings:
            holdings_data.append(
                {
                    "symbol": holding.symbol,
                    "units": holding.units,
                    "price": holding.price,
                }
            )

        # Use proper temporary file handling with permissions
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(holdings_data, f)
            f.flush()
            temp_file = f.name

        try:
            # Ensure file is readable
            os.chmod(temp_file, 0o666)

            result = capture_output(
                show_rebalancing_needs,
                holdings_file=temp_file,
                from_date=request.from_date,
                portfolio_size=request.size,
            )

            if result["status"] == "success":
                return JSONResponse(
                    content={
                        "status": "success",
                        "data": {
                            "holdings_count": len(request.holdings),
                            "size": request.size,
                            "from_date": request.from_date,
                            "output": result["output"],
                        },
                    }
                )
            else:
                raise HTTPException(status_code=500, detail=result)
        finally:
            # Always clean up the temporary file
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except (OSError, PermissionError):
                # Log but don't fail if we can't delete
                pass

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebalance/upload")
async def rebalance_portfolio_upload(
    file: UploadFile = File(...), from_date: str = Form(None), size: int = Form(5)
):
    """Analyze portfolio rebalancing needs using uploaded JSON/CSV file"""
    try:
        # Validate file type
        allowed_extensions = {".json", ".csv"}
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Please upload JSON or CSV file.",
            )

        # Save uploaded file temporarily with proper permissions
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            suffix=file_extension,
            delete=False,  # We'll delete manually after use
        ) as f:
            content = await file.read()
            f.write(content)
            f.flush()
            temp_file = f.name

        try:
            # Ensure file is readable
            os.chmod(temp_file, 0o666)

            result = capture_output(
                show_rebalancing_needs,
                holdings_file=temp_file,
                from_date=from_date,
                portfolio_size=size,
            )

            if result["status"] == "success":
                return JSONResponse(
                    content={
                        "status": "success",
                        "data": {
                            "filename": file.filename,
                            "file_type": file_extension,
                            "size": size,
                            "from_date": from_date,
                            "output": result["output"],
                        },
                    }
                )
            else:
                raise HTTPException(status_code=500, detail=result)
        finally:
            # Always clean up the temporary file
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except (OSError, PermissionError):
                # Log but don't fail if we can't delete
                pass

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/historical")
async def get_historical_portfolio_get(
    from_date: str = Query(description="Start date in YYYY-MM-DD format"),
    to_date: Optional[str] = Query(default=None, description="End date"),
    amount: float = Query(default=1000000, description="Investment amount"),
    size: int = Query(default=5, description="Portfolio size"),
):
    """Get historical portfolio analysis (GET method)"""
    try:
        result = capture_output(
            show_historical_portfolio,
            from_date_str=from_date,
            to_date_str=to_date,
            investment_amount=amount,
            portfolio_size=size,
        )

        if result["status"] == "success":
            return JSONResponse(
                content={
                    "status": "success",
                    "data": {
                        "from_date": from_date,
                        "to_date": to_date,
                        "amount": amount,
                        "size": size,
                        "output": result["output"],
                    },
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/historical")
async def get_historical_portfolio_post(request: HistoricalRequest):
    """Get historical portfolio analysis (POST method)"""
    try:
        result = capture_output(
            show_historical_portfolio,
            from_date_str=request.from_date,
            to_date_str=request.to_date,
            investment_amount=request.amount,
            portfolio_size=request.size,
        )

        if result["status"] == "success":
            return JSONResponse(
                content={
                    "status": "success",
                    "data": {
                        "from_date": request.from_date,
                        "to_date": request.to_date,
                        "amount": request.amount,
                        "size": request.size,
                        "output": result["output"],
                    },
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/app", response_class=HTMLResponse)
async def web_app():
    """Serve the web application"""
    return HTMLResponse(content=get_web_app_html())


def get_web_app_html():
    """Generate the HTML for the web application"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF Momentum Strategy</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card-shadow {
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
        .result-container {
            background: #f8fafc;
            border-left: 4px solid #3b82f6;
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    </style>
</head>
<body class="bg-gray-50" x-data="etfApp()">
    <!-- Header -->
    <div class="gradient-bg text-white py-8">
        <div class="container mx-auto px-4">
            <h1 class="text-4xl font-bold text-center mb-2">
                <i class="fas fa-chart-line mr-3"></i>
                ETF Momentum Strategy
            </h1>
            <p class="text-center text-blue-100 text-lg">
                Intelligent Portfolio Management with Momentum Analysis
            </p>
        </div>
    </div>

    <!-- Main Content -->
    <div class="container mx-auto px-4 py-8">
        <!-- Tab Navigation -->
        <div class="mb-8">
            <div class="flex flex-wrap justify-center space-x-1 bg-white rounded-lg p-1 card-shadow">
                <button @click="activeTab = 'portfolio'" 
                        :class="activeTab === 'portfolio' ? 'bg-blue-500 text-white' : 'text-gray-600 hover:text-blue-500'"
                        class="px-6 py-3 rounded-md font-medium transition-colors duration-200">
                    <i class="fas fa-briefcase mr-2"></i>Portfolio
                </button>
                <button @click="activeTab = 'rebalance'" 
                        :class="activeTab === 'rebalance' ? 'bg-blue-500 text-white' : 'text-gray-600 hover:text-blue-500'"
                        class="px-6 py-3 rounded-md font-medium transition-colors duration-200">
                    <i class="fas fa-balance-scale mr-2"></i>Rebalance
                </button>
                <button @click="activeTab = 'historical'" 
                        :class="activeTab === 'historical' ? 'bg-blue-500 text-white' : 'text-gray-600 hover:text-blue-500'"
                        class="px-6 py-3 rounded-md font-medium transition-colors duration-200">
                    <i class="fas fa-chart-area mr-2"></i>Historical
                </button>
            </div>
        </div>

        <!-- Portfolio Tab -->
        <div x-show="activeTab === 'portfolio'" class="space-y-6">
            <div class="bg-white rounded-lg card-shadow p-6">
                <h2 class="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-target text-blue-500 mr-3"></i>
                    Current Optimal Portfolio
                </h2>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            Investment Amount (₹)
                        </label>
                        <input type="number" x-model="portfolio.amount" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                               placeholder="1000000">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            Portfolio Size
                        </label>
                        <input type="number" x-model="portfolio.size" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                               placeholder="5">
                    </div>
                </div>
                
                <button @click="getPortfolio()" 
                        :disabled="loading.portfolio"
                        :class="loading.portfolio ? 'loading' : ''"
                        class="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-6 rounded-md transition-colors duration-200">
                    <i class="fas fa-rocket mr-2"></i>
                    <span x-text="loading.portfolio ? 'Analyzing...' : 'Get Optimal Portfolio'"></span>
                </button>
            </div>
            
            <div x-show="results.portfolio" class="bg-white rounded-lg card-shadow p-6 result-container">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">
                    <i class="fas fa-chart-pie text-green-500 mr-2"></i>Portfolio Results
                </h3>
                <pre x-text="results.portfolio" class="text-sm text-gray-700 font-mono"></pre>
            </div>
        </div>

        <!-- Rebalance Tab -->
        <div x-show="activeTab === 'rebalance'" class="space-y-6">
            <div class="bg-white rounded-lg card-shadow p-6">
                <h2 class="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-balance-scale text-blue-500 mr-3"></i>
                    Portfolio Rebalancing Analysis
                </h2>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            Holdings File (JSON/CSV)
                        </label>
                        <input type="file" @change="handleFileUpload($event)" 
                               accept=".json,.csv"
                               class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                        <p class="text-xs text-gray-500 mt-1">Upload your current holdings as JSON or CSV file</p>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            Purchase Date (Optional)
                        </label>
                        <input type="date" x-model="rebalance.fromDate" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            Target Portfolio Size
                        </label>
                        <input type="number" x-model="rebalance.size" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                               placeholder="5">
                    </div>
                </div>
                
                <button @click="analyzeRebalance()" 
                        :disabled="loading.rebalance || !rebalance.file"
                        :class="loading.rebalance ? 'loading' : ''"
                        class="w-full bg-green-500 hover:bg-green-600 text-white font-bold py-3 px-6 rounded-md transition-colors duration-200 disabled:bg-gray-400">
                    <i class="fas fa-calculator mr-2"></i>
                    <span x-text="loading.rebalance ? 'Analyzing...' : 'Analyze Rebalancing'"></span>
                </button>
            </div>
            
            <div x-show="results.rebalance" class="bg-white rounded-lg card-shadow p-6 result-container">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">
                    <i class="fas fa-exchange-alt text-yellow-500 mr-2"></i>Rebalancing Analysis
                </h3>
                <pre x-text="results.rebalance" class="text-sm text-gray-700 font-mono"></pre>
            </div>
        </div>

        <!-- Historical Tab -->
        <div x-show="activeTab === 'historical'" class="space-y-6">
            <div class="bg-white rounded-lg card-shadow p-6">
                <h2 class="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-chart-area text-blue-500 mr-3"></i>
                    Historical Portfolio Analysis
                </h2>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            From Date *
                        </label>
                        <input type="date" x-model="historical.fromDate" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            To Date (Optional)
                        </label>
                        <input type="date" x-model="historical.toDate" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            Investment Amount (₹)
                        </label>
                        <input type="number" x-model="historical.amount" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                               placeholder="1000000">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            Portfolio Size
                        </label>
                        <input type="number" x-model="historical.size" 
                               class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                               placeholder="5">
                    </div>
                </div>
                
                <button @click="getHistorical()" 
                        :disabled="loading.historical || !historical.fromDate"
                        :class="loading.historical ? 'loading' : ''"
                        class="w-full bg-purple-500 hover:bg-purple-600 text-white font-bold py-3 px-6 rounded-md transition-colors duration-200 disabled:bg-gray-400">
                    <i class="fas fa-history mr-2"></i>
                    <span x-text="loading.historical ? 'Analyzing...' : 'Get Historical Analysis'"></span>
                </button>
            </div>
            
            <div x-show="results.historical" class="bg-white rounded-lg card-shadow p-6 result-container">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">
                    <i class="fas fa-chart-line text-purple-500 mr-2"></i>Historical Analysis
                </h3>
                <pre x-text="results.historical" class="text-sm text-gray-700 font-mono"></pre>
            </div>
        </div>
    </div>

    <!-- Error Modal -->
    <div x-show="error" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div class="bg-white rounded-lg p-6 max-w-md mx-4">
            <h3 class="text-lg font-semibold text-red-600 mb-4">
                <i class="fas fa-exclamation-triangle mr-2"></i>Error
            </h3>
            <p x-text="error" class="text-gray-700 mb-4"></p>
            <button @click="error = null" class="w-full bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded">
                Close
            </button>
        </div>
    </div>

    <script>
        function etfApp() {
            return {
                activeTab: 'portfolio',
                loading: {
                    portfolio: false,
                    rebalance: false,
                    historical: false
                },
                results: {
                    portfolio: null,
                    rebalance: null,
                    historical: null
                },
                portfolio: {
                    amount: 1000000,
                    size: 5
                },
                rebalance: {
                    file: null,
                    fromDate: '',
                    size: 5
                },
                historical: {
                    fromDate: '',
                    toDate: '',
                    amount: 1000000,
                    size: 5
                },
                error: null,

                async getPortfolio() {
                    this.loading.portfolio = true;
                    this.results.portfolio = null;
                    
                    try {
                        const response = await fetch(`/portfolio?amount=${this.portfolio.amount}&size=${this.portfolio.size}`);
                        const data = await response.json();
                        
                        if (data.status === 'success') {
                            this.results.portfolio = data.data.output;
                        } else {
                            throw new Error(data.message || 'Failed to get portfolio');
                        }
                    } catch (err) {
                        this.error = err.message;
                    } finally {
                        this.loading.portfolio = false;
                    }
                },

                async analyzeRebalance() {
                    if (!this.rebalance.file) {
                        this.error = 'Please select a holdings file';
                        return;
                    }
                    
                    this.loading.rebalance = true;
                    this.results.rebalance = null;
                    
                    try {
                        const formData = new FormData();
                        formData.append('file', this.rebalance.file);
                        formData.append('size', this.rebalance.size);
                        if (this.rebalance.fromDate) {
                            formData.append('from_date', this.rebalance.fromDate);
                        }
                        
                        const response = await fetch('/rebalance/upload', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        if (data.status === 'success') {
                            this.results.rebalance = data.data.output;
                        } else {
                            throw new Error(data.message || 'Failed to analyze rebalance');
                        }
                    } catch (err) {
                        this.error = err.message;
                    } finally {
                        this.loading.rebalance = false;
                    }
                },

                async getHistorical() {
                    if (!this.historical.fromDate) {
                        this.error = 'Please select a from date';
                        return;
                    }
                    
                    this.loading.historical = true;
                    this.results.historical = null;
                    
                    try {
                        let url = `/historical?from_date=${this.historical.fromDate}&amount=${this.historical.amount}&size=${this.historical.size}`;
                        if (this.historical.toDate) {
                            url += `&to_date=${this.historical.toDate}`;
                        }
                        
                        const response = await fetch(url);
                        const data = await response.json();
                        
                        if (data.status === 'success') {
                            this.results.historical = data.data.output;
                        } else {
                            throw new Error(data.message || 'Failed to get historical analysis');
                        }
                    } catch (err) {
                        this.error = err.message;
                    } finally {
                        this.loading.historical = false;
                    }
                },

                handleFileUpload(event) {
                    const file = event.target.files[0];
                    if (file) {
                        // Validate file type
                        const allowedTypes = ['application/json', 'text/csv', 'application/vnd.ms-excel'];
                        const allowedExtensions = ['.json', '.csv'];
                        const extension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
                        
                        if (!allowedExtensions.includes(extension)) {
                            this.error = 'Please select a JSON or CSV file';
                            event.target.value = '';
                            return;
                        }
                        
                        this.rebalance.file = file;
                    }
                }
            }
        }
    </script>
</body>
</html>
    """


def cleanup_old_files():
    """Clean up old log files and any leftover temp files"""
    try:
        # Clean up old log files (keep only last 5)
        logs_dir = Path("logs")
        if logs_dir.exists():
            log_files = list(logs_dir.glob("momentum_strategy_*.log"))
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Keep only the 5 most recent log files
            for old_log in log_files[5:]:
                try:
                    old_log.unlink()
                    print(f"🗑️  Cleaned up old log file: {old_log.name}")
                except OSError:
                    pass

        # Clean up any temp files that might be left over
        temp_dir = Path(tempfile.gettempdir())
        for pattern in ["tmp*.json", "tmp*.csv"]:
            for temp_file in temp_dir.glob(pattern):
                try:
                    # Only delete files older than 1 hour
                    if datetime.now().timestamp() - temp_file.stat().st_mtime > 3600:
                        temp_file.unlink()
                        print(f"🗑️  Cleaned up old temp file: {temp_file.name}")
                except OSError:
                    pass

    except Exception as e:
        print(f"⚠️  Warning: Could not clean up old files: {e}")


def main():
    """Main entry point for the web server"""
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"🚀 Starting ETF Momentum Strategy Web Server")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🌐 Full address: http://{host}:{port}")
    print(f"🎨 Web App: http://{host}:{port}/app")
    print(f"📁 Working directory: {os.getcwd()}")
    print(
        f"🗂️  File logging: {'disabled' if os.environ.get('DISABLE_FILE_LOGGING') == '1' else 'enabled'}"
    )

    # Set uvicorn logger to warning level to reduce noise
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.WARNING)

    # Clean up old log and temp files
    cleanup_old_files()

    uvicorn.run(
        "web_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="warning",  # Reduce uvicorn log level
    )


if __name__ == "__main__":
    main()
