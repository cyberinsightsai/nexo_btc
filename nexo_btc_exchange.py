#!/usr/bin/env python3
"""
Nexo to Bitcoin Exchange Opportunity Finder

This script monitors the Nexo/BTC price ratio and identifies favorable moments
to exchange Nexo for Bitcoin based on local peaks in the ratio.
It also provides forecasts for future price ratios using Prophet.
"""

import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional
import os
import sys
import argparse
from prophet import Prophet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nexo_btc_exchange.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("NexoBTCExchange")

class NexoBTCExchangeMonitor:
    """Monitor Nexo/BTC price ratio and identify exchange opportunities."""
    
    def __init__(
        self,
        exchange_id: str = 'binance',
        nexo_symbol: str = 'NEXO/USDT',
        btc_symbol: str = 'BTC/USDT',
        timeframe: str = '12h',
        window_size: int = 168,  # 1 week worth of hourly data
        peak_threshold: float = 0.9,  # 90th percentile
        data_dir: str = './data'
    ):
        """
        Initialize the exchange monitor.
        
        Args:
            exchange_id: CCXT exchange ID to use
            nexo_symbol: Nexo trading pair symbol
            btc_symbol: Bitcoin trading pair symbol
            timeframe: Timeframe for historical data
            window_size: Number of periods to analyze for trends
            peak_threshold: Percentile threshold to identify peaks (0.0-1.0)
            data_dir: Directory to save data
        """
        self.exchange_id = exchange_id
        self.nexo_symbol = nexo_symbol
        self.btc_symbol = btc_symbol
        self.timeframe = timeframe
        self.window_size = window_size
        self.peak_threshold = peak_threshold
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize exchange
        self.exchange = self._init_exchange()
        
        # Initialize historical data storage
        self.history_df = self._init_history_df()
        
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize and return CCXT exchange instance."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
            })
            logger.info(f"Successfully initialized {self.exchange_id} exchange")
            return exchange
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def _init_history_df(self) -> pd.DataFrame:
        """Initialize or load historical data DataFrame."""
        history_file = os.path.join(self.data_dir, f"nexo_btc_history_{self.timeframe}.csv")
        
        if os.path.exists(history_file):
            try:
                df = pd.read_csv(history_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"Loaded historical data from {history_file}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load historical data: {e}. Creating new dataset.")
        
        # Create empty DataFrame
        return pd.DataFrame(columns=[
            'timestamp', 'nexo_price', 'btc_price', 'nexo_btc_ratio',
            'ratio_sma', 'ratio_percentile', 'is_peak'
        ])
    
    def fetch_current_prices(self) -> Tuple[float, float]:
        """Fetch current prices for Nexo and BTC."""
        try:
            nexo_ticker = self.exchange.fetch_ticker(self.nexo_symbol)
            btc_ticker = self.exchange.fetch_ticker(self.btc_symbol)
            
            nexo_price = nexo_ticker['last']
            btc_price = btc_ticker['last']
            
            logger.info(f"Current prices - NEXO: ${nexo_price:.4f}, BTC: ${btc_price:.2f}")
            return nexo_price, btc_price
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            raise
    
    def fetch_historical_data(self, days_back: int = 90) -> pd.DataFrame:
        """
        Fetch historical price data for both assets.
        
        Args:
            days_back: Number of days of historical data to fetch. Default is 90 (3 months).
            
        Returns:
            DataFrame with historical price data
        """
        try:
            # Calculate start time - limit to 90 days (3 months)
            days_back = min(days_back, 90)  # Enforce 3-month limit
            since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            
            # Fetch OHLCV data
            nexo_ohlcv = self.exchange.fetch_ohlcv(self.nexo_symbol, self.timeframe, since)
            btc_ohlcv = self.exchange.fetch_ohlcv(self.btc_symbol, self.timeframe, since)
            
            # Convert to DataFrames
            nexo_df = pd.DataFrame(nexo_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamps to datetime
            nexo_df['timestamp'] = pd.to_datetime(nexo_df['timestamp'], unit='ms')
            btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
            
            # Merge DataFrames
            merged_df = pd.merge(
                nexo_df[['timestamp', 'close']].rename(columns={'close': 'nexo_price'}),
                btc_df[['timestamp', 'close']].rename(columns={'close': 'btc_price'}),
                on='timestamp'
            )
            
            # Calculate ratio
            merged_df['nexo_btc_ratio'] = merged_df['nexo_price'] / merged_df['btc_price']
            
            logger.info(f"Fetched historical data: {len(merged_df)} entries (last {days_back} days)")
            return merged_df
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def update_historical_data(self) -> None:
        """Update historical data with latest values, fetching from last recorded date until today."""
        try:
            # Get the last recorded date from history_df
            if not self.history_df.empty:
                last_date = self.history_df['timestamp'].max()
                days_since_last = (datetime.now() - last_date).days
                logger.info(f"Last recorded data: {last_date.strftime('%Y-%m-%d %H:%M:%S')}, {days_since_last} days ago")
                
                # If last data is relatively recent (within a day), just add current price
                if days_since_last < 1:
                    # Get current prices only
                    nexo_price, btc_price = self.fetch_current_prices()
                    
                    # Create new entry
                    new_entry = pd.DataFrame([{
                        'timestamp': datetime.now(),
                        'nexo_price': nexo_price,
                        'btc_price': btc_price,
                        'nexo_btc_ratio': nexo_price / btc_price
                    }])
                    
                    # Append to history
                    self.history_df = pd.concat([self.history_df, new_entry], ignore_index=True)
                else:
                    # Fetch historical data for missing period (max 90 days)
                    days_to_fetch = min(days_since_last + 1, 90)
                    logger.info(f"Fetching {days_to_fetch} days of missing historical data")
                    
                    new_data = self.fetch_historical_data(days_back=days_to_fetch)
                    
                    # Filter to only get new data
                    new_data = new_data[new_data['timestamp'] > last_date]
                    
                    # Merge with existing data
                    if not new_data.empty:
                        logger.info(f"Adding {len(new_data)} new data points from {new_data['timestamp'].min().strftime('%Y-%m-%d')} to {new_data['timestamp'].max().strftime('%Y-%m-%d')}")
                        self.history_df = pd.concat([self.history_df, new_data], ignore_index=True)
                        self.history_df = self.history_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            else:
                # Empty DataFrame, fetch last 90 days
                logger.info("No existing data, fetching 90 days of historical data")
                new_data = self.fetch_historical_data(days_back=90)
                self.history_df = new_data
            
            # Trim data to keep only last 90 days
            cutoff_date = datetime.now() - timedelta(days=90)
            self.history_df = self.history_df[self.history_df['timestamp'] >= cutoff_date]
            
            # Calculate analytics
            self._calculate_analytics()
            
            # Save updated data
            self._save_history()
            
            logger.info(f"Data updated successfully. Range: {self.history_df['timestamp'].min().strftime('%Y-%m-%d')} to {self.history_df['timestamp'].max().strftime('%Y-%m-%d')}")
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")
            raise
    
    def _calculate_analytics(self) -> None:
        """Calculate analytics on the historical data."""
        df = self.history_df.copy()
        
        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate moving average
        df['ratio_sma'] = df['nexo_btc_ratio'].rolling(window=self.window_size, min_periods=1).mean()
        
        # Calculate percentile within recent window
        df['ratio_percentile'] = df['nexo_btc_ratio'].rolling(window=self.window_size, min_periods=1).apply(
            lambda x: np.percentile(np.nan_to_num(x), 100, interpolation='linear') 
            if np.isnan(x).all() else 
            percentileofscore(x.dropna(), x.iloc[-1])
        )
        
        # Identify peaks (local maxima within the window that exceed threshold)
        df['is_peak'] = False
        
        # Find local peaks within the window size
        for i in range(self.window_size, len(df)):
            window = df['nexo_btc_ratio'].iloc[i-self.window_size:i+1]
            if window.iloc[-1] > window.iloc[:-1].quantile(self.peak_threshold):
                df.at[df.index[i], 'is_peak'] = True
        
        self.history_df = df
    
    def _save_history(self) -> None:
        """Save historical data to CSV file."""
        history_file = os.path.join(self.data_dir, f"nexo_btc_history_{self.timeframe}.csv")
        self.history_df.to_csv(history_file, index=False)
        logger.info(f"Saved historical data to {history_file}")
    
    def check_exchange_opportunity(self) -> Optional[Dict]:
        """Check if current market conditions present a good exchange opportunity."""
        if len(self.history_df) < self.window_size:
            logger.warning("Not enough historical data to evaluate opportunity")
            return None
        
        # Get latest data point
        latest = self.history_df.iloc[-1]
        
        # Calculate percentile in the recent window
        recent_window = self.history_df['nexo_btc_ratio'].iloc[-self.window_size:]
        current_percentile = percentileofscore(recent_window.dropna(), recent_window.iloc[-1])
        
        # Check if we're at or near a peak
        is_opportunity = False
        confidence = 0.0
        
        if latest['is_peak']:
            is_opportunity = True
            confidence = 0.9
        elif current_percentile > self.peak_threshold * 100:
            is_opportunity = True
            confidence = 0.7 + (current_percentile - self.peak_threshold * 100) / (100 - self.peak_threshold * 100) * 0.2
        
        if is_opportunity:
            return {
                'timestamp': latest['timestamp'],
                'nexo_price': latest['nexo_price'],
                'btc_price': latest['btc_price'],
                'nexo_btc_ratio': latest['nexo_btc_ratio'],
                'percentile': current_percentile,
                'confidence': confidence,
                'message': f"Good opportunity to exchange NEXO for BTC (confidence: {confidence:.2f})"
            }
        
        return None
    
    def predict_future_ratio(self, days_ahead: int = 3) -> pd.DataFrame:
        """
        Predict future NEXO/BTC ratio using Prophet.
        
        Args:
            days_ahead: Number of days to predict into the future
            
        Returns:
            DataFrame with forecasted values
        """
        if len(self.history_df) < 10:
            logger.warning("Not enough historical data for prediction")
            return pd.DataFrame()
            
        try:
            # Prepare data for Prophet
            prophet_df = self.history_df[['timestamp', 'nexo_btc_ratio']].copy()
            prophet_df.columns = ['ds', 'y']  # Prophet requires these column names
            
            # Create and fit model
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=False,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=days_ahead * 24, freq='H')
            
            # Make prediction
            forecast = model.predict(future)
            
            # Extract relevant columns
            forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_result.columns = ['timestamp', 'predicted_ratio', 'lower_bound', 'upper_bound']
            
            # Only keep future predictions
            last_timestamp = self.history_df['timestamp'].max()
            future_forecast = forecast_result[forecast_result['timestamp'] > last_timestamp]
            
            logger.info(f"Generated {len(future_forecast)} future predictions for {days_ahead} days ahead")
            return future_forecast
            
        except Exception as e:
            logger.error(f"Error predicting future values: {e}")
            return pd.DataFrame()

    def plot_ratio_history(self, save_path: Optional[str] = None, include_prediction: bool = True) -> None:
        """
        Plot the history of the NEXO/BTC ratio with peaks highlighted and optional predictions.
        
        Args:
            save_path: Path to save the plot image
            include_prediction: Whether to include future predictions
        """
        if len(self.history_df) < 2:
            logger.warning("Not enough data to generate plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot ratio
        plt.plot(self.history_df['timestamp'], self.history_df['nexo_btc_ratio'], label='NEXO/BTC Ratio')
        
        # Plot moving average
        plt.plot(self.history_df['timestamp'], self.history_df['ratio_sma'], 
                 label=f'SMA ({self.window_size} periods)', linestyle='--')
        
        # Highlight peaks
        peaks = self.history_df[self.history_df['is_peak']]
        if not peaks.empty:
            plt.scatter(peaks['timestamp'], peaks['nexo_btc_ratio'], 
                        color='red', marker='^', s=100, label='Exchange Opportunity')
        
        # Add predictions if requested
        if include_prediction:
            predictions = self.predict_future_ratio(days_ahead=3)
            if not predictions.empty:
                # Plot predictions
                plt.plot(predictions['timestamp'], predictions['predicted_ratio'], 
                         label='3-Day Forecast', color='green', linestyle='-.')
                
                # Plot prediction bounds
                plt.fill_between(
                    predictions['timestamp'],
                    predictions['lower_bound'],
                    predictions['upper_bound'],
                    color='green', alpha=0.2, label='Prediction Interval'
                )
        
        # Formatting
        plt.title('NEXO/BTC Price Ratio History and Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price Ratio')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def run_monitor(self, interval_minutes: int = 60, run_forever: bool = False) -> None:
        """
        Run the exchange monitor once or continuously.
        
        Args:
            interval_minutes: Minutes between checks (only relevant if run_forever=True)
            run_forever: Whether to run indefinitely (False by default)
        """
        try:
            iteration = 0
            
            while True:
                iteration += 1
                logger.info(f"Starting monitoring iteration {iteration}")
                
                # Update data from last recorded date to today
                self.update_historical_data()
                
                # Check for opportunity
                opportunity = self.check_exchange_opportunity()
                
                # Generate predictions for the next 3 days
                predictions = self.predict_future_ratio(days_ahead=3)
                if not predictions.empty:
                    logger.info("--- 3-Day Forecast ---")
                    # Group by day for readability
                    daily_pred = predictions.set_index('timestamp').resample('D').mean().reset_index()
                    for _, row in daily_pred.iterrows():
                        date_str = row['timestamp'].strftime('%Y-%m-%d')
                        logger.info(f"Date: {date_str}, Predicted Ratio: {row['predicted_ratio']:.8f} " +
                                   f"(Range: {row['lower_bound']:.8f} to {row['upper_bound']:.8f})")
                
                if opportunity:
                    logger.info(f"OPPORTUNITY DETECTED: {opportunity['message']}")
                    logger.info(f"Current NEXO/BTC ratio: {opportunity['nexo_btc_ratio']:.8f}")
                    logger.info(f"Percentile: {opportunity['percentile']:.2f}%")
                    
                    # Generate plot
                    plot_path = os.path.join(self.data_dir, f"nexo_btc_opportunity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    self.plot_ratio_history(save_path=plot_path, include_prediction=True)
                else:
                    logger.info("No favorable exchange opportunity detected at this time")
                    
                    # Always generate a plot
                    plot_path = os.path.join(self.data_dir, f"nexo_btc_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    self.plot_ratio_history(save_path=plot_path, include_prediction=True)
                
                # By default, exit after one iteration (run_forever is False by default)
                if not run_forever:
                    logger.info("Single-run mode, exiting")
                    if opportunity:
                        print("GOOD OPPORTUNITY - Exchange NEXO for BTC now!")
                    else:
                        print("NO OPPORTUNITY - Wait for better conditions")
                    break
                
                # Only reached if run_forever=True
                logger.info(f"Continuous mode: Sleeping for {interval_minutes} minutes until next check")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            raise


def percentileofscore(a, score):
    """
    Calculate the percentile rank of a score relative to a list of scores.
    
    This function is a simplified version of scipy.stats.percentileofscore
    for use when scipy is not available.
    
    Args:
        a: Array of scores to find the percentile in
        score: Score to find the percentile of
        
    Returns:
        The percentile rank (0-100) of the score
    """
    a = np.asarray(a)
    n = len(a)
    
    if n == 0:
        return np.nan
        
    # Number of scores below score
    below = np.sum(a < score)
    # Number of scores equal to score
    equal = np.sum(a == score)
    
    # Linear interpolation between points
    if equal:
        return 100.0 * (below + 0.5 * equal) / n
    else:
        return 100.0 * below / n


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NEXO to BTC Exchange Opportunity Monitor')
    
    parser.add_argument('--exchange', type=str, default='binance',
                        help='CCXT exchange ID (default: binance)')
    parser.add_argument('--nexo-symbol', type=str, default='NEXO/USDT',
                        help='NEXO trading pair (default: NEXO/USDT)')
    parser.add_argument('--btc-symbol', type=str, default='BTC/USDT',
                        help='BTC trading pair (default: BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Timeframe for price data (default: 1h)')
    parser.add_argument('--window', type=int, default=168,
                        help='Window size for analysis in periods (default: 168 = 1 week of hourly data)')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Percentile threshold for opportunity detection (default: 0.9)')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuously with interval checks (default: run once and exit)')
    parser.add_argument('--interval', type=int, default=60,
                        help='Check interval in minutes for continuous mode (default: 60)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Generate plot from existing data and exit')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to store data (default: ./data)')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Create monitor
        monitor = NexoBTCExchangeMonitor(
            exchange_id=args.exchange,
            nexo_symbol=args.nexo_symbol,
            btc_symbol=args.btc_symbol,
            timeframe=args.timeframe,
            window_size=args.window,
            peak_threshold=args.threshold,
            data_dir=args.data_dir
        )
        
        if args.plot_only:
            logger.info("Generating plot from existing data")
            monitor.plot_ratio_history()
        else:
            # Run monitor - by default it runs once and exits
            logger.info(f"Running in {'continuous' if args.continuous else 'single-run'} mode")
            monitor.run_monitor(
                interval_minutes=args.interval,
                run_forever=args.continuous
            )
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required packages: pip install ccxt pandas numpy matplotlib prophet")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
