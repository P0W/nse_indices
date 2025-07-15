# Creating Your Own Strategy

This guide explains how to create and add your own trading strategy to the framework.

## Overview

Each strategy consists of two main classes:
1. A strategy class that inherits from `BaseStrategy`
2. A configuration class that inherits from `StrategyConfig`

## Step-by-Step Guide

### 1. Create a Strategy File

Create a new Python file in the `strategies/` folder (e.g., `my_strategy.py`):

```python
from .base_strategy import BaseStrategy, StrategyConfig
import backtrader as bt
from typing import Dict, Any, List

class MyTradingStrategy(BaseStrategy):
    """Your custom trading strategy"""
    
    params = (
        ('param1', 10),
        ('param2', 2.0),
        ('printlog', False),
    )
    
    def __init__(self):
        super().__init__()
        # Initialize your indicators here
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.param1)
    
    def next(self):
        # Increment bar counter (required for initialization)
        self.bar_executed += 1
        
        # Skip until we have enough data
        if self.bar_executed < self.p.param1:
            return
        
        # Your trading logic
        if self.data.close[0] > self.sma[0]:
            if not self.getposition():
                self.buy()
                self.log(f"BUY at {self.data.close[0]:.2f}")
        elif self.data.close[0] < self.sma[0]:
            if self.getposition():
                self.sell()
                self.log(f"SELL at {self.data.close[0]:.2f}")
        
        # Track portfolio value
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date(0))

class MyStrategyConfig(StrategyConfig):
    """Configuration for your strategy"""
    
    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Define parameters to optimize"""
        return {
            "param1": [5, 10, 15, 20],
            "param2": [1.5, 2.0, 2.5, 3.0],
            "printlog": [False],
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        """Define default parameters"""
        return {"param1": 10, "param2": 2.0, "printlog": False}
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters"""
        return params.get("param1", 0) > 0 and params.get("param2", 0) > 0
    
    def get_strategy_class(self) -> type:
        """Return the strategy class"""
        return MyTradingStrategy
```

### 2. Update `__init__.py`

Add your strategy to `strategies/__init__.py`:

```python
from .my_strategy import MyTradingStrategy, MyStrategyConfig

__all__ = [
    # ... existing exports
    'MyTradingStrategy',
    'MyStrategyConfig',
]
```

### 3. Register in `unified_runner.py`

Add your strategy to `STRATEGY_REGISTRY` in `unified_runner.py`:

```python
STRATEGY_REGISTRY = {
    # ... existing strategies
    'my_strategy': {
        'name': 'My Trading Strategy',
        'description': 'Description of what your strategy does',
        'config_class': MyStrategyConfig,
    },
}
```

### 4. Run Your Strategy

```bash
# Basic backtest
python unified_runner.py --strategy my_strategy

# With optimization
python unified_runner.py --strategy my_strategy --optimize

# With specific universe
python unified_runner.py --strategy my_strategy --universe nifty100
```

## Essential Components

### Strategy Class
Your strategy class must:
- Inherit from `BaseStrategy`
- Define parameters in the `params` tuple
- Initialize with `super().__init__()`
- Implement trading logic in the `next()` method
- Track entry/exit conditions
- Include `bar_executed` counter
- Use `self.log()` for logging (provided by BaseStrategy)

### Config Class
Your config class must:
- Inherit from `StrategyConfig`
- Implement `get_parameter_grid()` for optimization parameters
- Implement `get_default_params()` for default values
- Implement `validate_params()` to check parameter validity
- Implement `get_strategy_class()` to link to your strategy class

## Key Points to Remember

1. **Initialization**: Always include `self.bar_executed = 0` in your strategy
2. **Wait for Data**: Skip execution until you have enough data points
3. **Portfolio Tracking**: Update `self.portfolio_values` and `self.dates` for performance tracking
4. **Logging**: Use `self.log()` for debugging (e.g., `self.log("Buy signal")`)
5. **Parameter Types**: Keep parameters simple (integers, floats, booleans)

## Example Strategies

For more complex examples, check the existing strategies:
- `momentum_strategy.py` - Uses multiple indicators and rebalances portfolio
- `pairs_strategy.py` - Trading correlated pairs
- `pmv_momentum_strategy.py` - P=MV momentum with weekly exits
