# Trading Strategies Framework

A unified framework for developing and testing trading strategies with a clean, modular architecture.

## 📁 Project Structure

```
.
├── strategies/                    # All trading strategies
│   ├── __init__.py               # Package exports
│   ├── base_strategy.py          # Base classes for all strategies
│   ├── momentum_strategy.py      # Adaptive momentum strategy
│   ├── pairs_strategy.py         # Pairs trading strategy
│   ├── mean_reversion_strategy.py # Portfolio mean reversion
│   └── statistical_trend_strategy.py # Multi-indicator trend following
├── experiment_framework.py       # Unified experiment engine
├── unified_runner.py            # Main script to run any strategy
├── run_experiments.py           # Interactive experiment runner
├── utils.py                     # Market data loader and utilities
└── README.md                    # This file
```

## 🚀 Quick Start

### List Available Strategies
```bash
python unified_runner.py --list
```

### Run a Strategy Backtest
```bash
# Run momentum strategy with default parameters
python unified_runner.py --strategy momentum

# Run pairs trading with specific symbols
python unified_runner.py --strategy pairs --symbols HDFCBANK.NS ICICIBANK.NS

# Run with custom date range
python unified_runner.py --strategy mean_reversion --start-date 2020-01-01 --end-date 2023-12-31
```

### Run Parameter Optimization
```bash
# Optimize momentum strategy
python unified_runner.py --strategy momentum --optimize --max-experiments 50

# Optimize pairs trading
python unified_runner.py --strategy pairs --optimize --symbols HDFCBANK.NS ICICIBANK.NS
```

### Interactive Mode
```bash
python unified_runner.py
```

## 📊 Available Strategies

### 1. Adaptive Momentum Strategy
- **Key**: `momentum`
- **Description**: Multi-stock momentum-based trend following
- **Features**: Dynamic stock selection, rebalancing, momentum indicators
- **Required Data**: Multiple stocks (5+ recommended)

### 2. Pairs Trading Strategy  
- **Key**: `pairs`
- **Description**: Statistical arbitrage for correlated pairs
- **Features**: Beta calculation, spread analysis, mean reversion
- **Required Data**: Exactly 2 correlated stocks

### 3. Portfolio Mean Reversion Strategy
- **Key**: `mean_reversion`
- **Description**: Multi-stock mean reversion based on z-scores
- **Features**: Statistical overbought/oversold conditions
- **Required Data**: Multiple stocks (3+ recommended)

### 4. Statistical Trend Following Strategy
- **Key**: `statistical_trend`
- **Description**: Multi-indicator trend following system
- **Features**: MACD, RSI, Bollinger Bands, linear regression
- **Required Data**: Multiple stocks (5+ recommended)

## 🔧 Adding New Strategies

To add a new trading strategy, follow these steps:

### 1. Create Strategy File
Create a new file in the `strategies/` folder (e.g., `my_strategy.py`):

```python
from .base_strategy import BaseStrategy, StrategyConfig
import backtrader as bt
from typing import Dict, Any, List

class MyTradingStrategy(BaseStrategy):
    \"\"\"Your custom trading strategy\"\"\"
    
    params = (
        ('param1', 10),
        ('param2', 2.0),
    )
    
    def __init__(self):
        super().__init__()
        # Initialize your indicators here
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.param1)
    
    def execute_strategy(self):
        \"\"\"Implement your trading logic here\"\"\"
        # Your trading logic
        if self.data.close[0] > self.sma[0]:
            if not self.getposition():
                self.buy()
        elif self.data.close[0] < self.sma[0]:
            if self.getposition():
                self.sell()

class MyStrategyConfig(StrategyConfig):
    \"\"\"Configuration for your strategy\"\"\"
    
    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "param1": [5, 10, 15, 20],
            "param2": [1.5, 2.0, 2.5, 3.0],
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        return {"param1": 10, "param2": 2.0}
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        return all(v > 0 for v in params.values())
    
    def get_strategy_class(self) -> type:
        return MyTradingStrategy
    
    def get_required_data_feeds(self) -> int:
        return 1  # Single stock strategy
```

### 2. Update Package Exports
Add your strategy to `strategies/__init__.py`:

```python
from .my_strategy import MyTradingStrategy, MyStrategyConfig

__all__ = [
    # ... existing exports
    'MyTradingStrategy',
    'MyStrategyConfig',
]
```

### 3. Register Strategy
Add your strategy to `STRATEGY_REGISTRY` in `unified_runner.py`:

```python
STRATEGY_REGISTRY = {
    # ... existing strategies
    'my_strategy': {
        'name': 'My Trading Strategy',
        'description': 'Description of what your strategy does',
        'config_class': MyStrategyConfig,
        'requires_multiple_stocks': False,  # or True
    },
}
```

### 4. Test Your Strategy
```bash
python unified_runner.py --strategy my_strategy
```

## 🏗️ Framework Architecture

### Base Classes

#### BaseStrategy
All trading strategies inherit from `BaseStrategy` which provides:
- Portfolio performance tracking
- Trade statistics
- Common utility methods
- Abstract `execute_strategy()` method to implement

#### StrategyConfig  
All strategy configurations inherit from `StrategyConfig` which provides:
- Parameter grid definition for optimization
- Parameter validation
- Strategy class reference
- Composite score calculation

### Experiment Framework
The `UnifiedExperimentFramework` class provides:
- Parameter optimization with parallel processing
- Single experiment execution
- Results analysis and visualization
- Progress tracking

## 🔧 Configuration

### Default Parameters
Each strategy defines its own default parameters in the config class.

### Parameter Optimization
The framework automatically generates parameter combinations from the parameter grid and tests them in parallel.

### Data Requirements
- Strategies specify their data feed requirements
- Automatic validation ensures correct number of stocks
- Support for single stock, pairs, and multi-stock strategies

## 📈 Performance Metrics

All strategies track:
- **Total Return**: Absolute percentage return
- **Sharpe Ratio**: Risk-adjusted return measure  
- **Maximum Drawdown**: Worst peak-to-trough loss
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of total profits to total losses

## 🛠️ Utilities

### Market Data Loader
- Automatic data fetching and caching
- Support for NSE Indian stocks
- Parallel data loading for multiple symbols
- Smart caching to avoid redundant API calls

### Commission Model
- Realistic Indian brokerage commission structure
- SEBI compliant cost modeling
- Includes brokerage, STT, GST, and regulatory charges

## 📊 Visualization

The framework provides comprehensive visualization:
- Portfolio growth charts
- Drawdown analysis
- Monthly returns heatmap
- Performance metrics dashboard
- Parameter optimization results

## 🚀 Performance Tips

1. **Parallel Processing**: Use `--max-workers` for faster optimization
2. **Data Caching**: Data is automatically cached to avoid re-downloading
3. **Parameter Limits**: Reasonable parameter grids for faster optimization
4. **Memory Management**: Framework handles large datasets efficiently

## 🐛 Troubleshooting

### Common Issues

1. **ImportError**: Make sure you're running from the project root directory
2. **Data Issues**: Check internet connection for market data fetching
3. **Memory Issues**: Reduce number of stocks or date range for large backtests
4. **Parameter Validation**: Check that your parameter combinations are valid

### Debug Mode
Enable debug logging by setting `printlog=True` in strategy parameters.

## 📚 Examples

See the existing strategies in the `strategies/` folder for complete examples of:
- Single stock strategies (statistical trend)
- Pairs trading strategies (pairs)
- Multi-stock strategies (momentum, mean reversion)

## 🤝 Contributing

1. Follow the base class structure
2. Add comprehensive docstrings
3. Include parameter validation
4. Test your strategy thoroughly
5. Update documentation

## 📄 License

This project is for educational and research purposes.
