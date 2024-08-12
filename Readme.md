[![Auto Run and Push](https://github.com/P0W/nse_indices/actions/workflows/main.yml/badge.svg)](https://github.com/P0W/nse_indices/actions/workflows/main.yml)

# NSE indices analyzer and builder

* [Thematic Indices](https://niftyindices.com/indices/equity/thematic-indices)
* [Strategy Indices](https://niftyindices.com/indices/equity/strategy-indices)

- Particularly analyzing [NSE INDIA DEFENCE Index](https://niftyindices.com/indices/equity/thematic-indices/nifty-india-defence) and [NSE SME EMERGE Index](https://niftyindices.com/indices/equity/thematic-indices/nifty-sme-emerge)
- Above two indices have given an _**astonishing CAGR ~60.0 %**_ over five years.
- NSE India Defence Index is rebalance semi-annually
- NSE Sme Emerge Index is rebalanced quaterly
- Top 10-15 stocks can potentialy be deployed via smallcase or brokers basket.

---

## Setup

1. **Clone the repository:**
	```sh
	git clone <repository-url>
	```

2. **Navigate to the project directory:**
	```sh
	cd nse_indices
	```

3. **Create a virtual environment:**
	```sh
	python -m venv venv
	```

4. **Activate the virtual environment:**
	- On Windows:
		```sh
		.\Scripts\activate
		```
	- On Unix or MacOS:
		```sh
		source Scripts/activate
		```

5. **Install the dependencies:**
	```sh
	pip install -r requirements.txt
	```

## Usage

Run the main script:
```sh
python main.py
```

## Last Run data
CSVs and JSON can be found [here](https://github.com/P0W/nse_indices/tree/main/data)

<sub>_Special thanks to [finology](https://ticker.finology.in/) and [nseindia](https://niftyindices.com/) for data_</sub>
