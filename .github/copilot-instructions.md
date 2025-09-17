# Copilot Instructions for NEXO to BTC Exchange Monitor

## Project Overview
This project monitors the NEXO/BTC price ratio to identify optimal moments for exchanging NEXO for Bitcoin. It fetches price data from crypto exchanges, analyzes local peaks, and generates alerts and visualizations for trading opportunities.

## Key Files & Structure
- `nexo_btc_exchange.py`: Main script for monitoring, analysis, and plotting. All core logic is here.
- `Nexo_BTC_Exchange_Monitor.ipynb`: Jupyter notebook for exploration, prototyping, or visualization.
- `data/`: Stores historical price data and generated plots.
- `requirements.txt`: Lists required Python packages.

## Developer Workflows
- **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
- **Run the monitor:**
  ```bash
  python nexo_btc_exchange.py [options]
  ```
  See `README.md` for all CLI options (e.g., `--exchange`, `--window`, `--interval`, `--plot-only`).
- **Generate plots only:**
  ```bash
  python nexo_btc_exchange.py --plot-only
  ```
- **Jupyter analysis:**
  Use the notebook for ad-hoc analysis or visualization.

## Patterns & Conventions
- All persistent data (CSV, PNG) is stored in `data/`.
- Uses [ccxt](https://github.com/ccxt/ccxt) for exchange API access.
- CLI options are handled in `nexo_btc_exchange.py` (argparse or similar).
- Plots are timestamped and named as `nexo_btc_forecast_YYYYMMDD_HHMMSS.png`.
- Logging is written to `nexo_btc_exchange.log`.

## Integration & Extensibility
- To add new exchanges or symbols, use the CLI options—no code changes needed for supported exchanges.
- For new analysis logic, extend `nexo_btc_exchange.py` and follow the pattern of reading from/writing to `data/`.

## Examples
- Run with custom window:
  ```bash
  python nexo_btc_exchange.py --window 96
  ```
- Plot from existing data:
  ```bash
  python nexo_btc_exchange.py --plot-only
  ```

## Reference
- See `README.md` for full usage and option details.
- See `data/README.md` for data storage notes.

---
*Update this file if you add new workflows, conventions, or major features.*
