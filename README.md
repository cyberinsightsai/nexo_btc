# NEXO to BTC Exchange Opportunity Monitor

This tool monitors the NEXO/BTC price ratio and identifies favorable moments to exchange NEXO for Bitcoin based on local peaks in the ratio.

## Features

- Tracks NEXO and BTC prices using cryptocurrency exchange APIs
- Calculates the NEXO/BTC price ratio over time
- Identifies local peaks in the ratio
- Generates alerts when a good exchange opportunity is detected
- Creates visualizations of price ratio history with highlighted opportunities

## Requirements

```
pip install ccxt pandas numpy matplotlib
```

## Usage

Basic usage:

```bash
python nexo_btc_exchange.py
```

This will run the monitor continuously with default settings.

### Command Line Options

```
--exchange        CCXT exchange ID (default: binance)
--nexo-symbol     NEXO trading pair (default: NEXO/USDT)
--btc-symbol      BTC trading pair (default: BTC/USDT)
--timeframe       Timeframe for price data (default: 1h)
--window          Window size for analysis in periods (default: 168 = 1 week of hourly data)
--threshold       Percentile threshold for opportunity detection (default: 0.9)
--interval        Check interval in minutes (default: 60)
--run-once        Run once and exit (default: run continuously)
--plot-only       Generate plot from existing data and exit
--data-dir        Directory to store data (default: ./data)
```

### Examples

Run with a different exchange:
```bash
python nexo_btc_exchange.py --exchange kucoin
```

Check more frequently:
```bash
python nexo_btc_exchange.py --interval 15
```

Use a shorter timeframe and analysis window:
```bash
python nexo_btc_exchange.py --timeframe 15m --window 96
```

Just generate a plot from existing data:
```bash
python nexo_btc_exchange.py --plot-only
```

## How It Works

The tool identifies good exchange opportunities by:

1. Collecting historical price data for both NEXO and BTC
2. Calculating the NEXO/BTC price ratio over time
3. Using a sliding window to identify when the current ratio is at a local peak
4. Calculating the percentile rank of the current ratio within the recent window
5. Alerting when the ratio is above a threshold percentile (default: 90th percentile)

## Data Storage

All historical data is stored in CSV files in the data directory. This allows the tool to resume monitoring with existing data when restarted.

## Visualization

The tool can generate plots showing the NEXO/BTC ratio over time, with highlighted exchange opportunities. When an opportunity is detected, a plot is automatically saved to the data directory.