#!/usr/bin/env python3
"""
MT5 Historical Data Downloader
=============================
Download multi-timeframe historical data for backtesting
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from colorama import init, Fore, Style
init(autoreset=True)

class MT5DataDownloader:
    """Download historical data from MT5."""
    
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol
        self.connected = False
        
    def connect(self):
        """Connect to MT5."""
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Failed to get account info")
            return False
            
        print(f"✅ Connected to MT5 - Account: {account_info.login}")
        
        # Ensure symbol is selected
        if not mt5.symbol_select(self.symbol, True):
            print(f"❌ Failed to select symbol {self.symbol}: {mt5.last_error()}")
            return False
        
        print(f"✅ Symbol {self.symbol} selected")
        self.connected = True
        return True
    
    def download_timeframe_data(self, timeframe_str, timeframe_mt5, start_date, end_date):
        """Download data for specific timeframe with custom date range."""
        if not self.connected:
            print("❌ Not connected to MT5")
            return None
        
        print(f"\n📊 Downloading {timeframe_str} data...")
        print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # Get historical data
            rates = mt5.copy_rates_range(self.symbol, timeframe_mt5, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                print(f"❌ No data available for {timeframe_str}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['Gmt time'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename columns to match training format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            })
            
            # Reorder columns
            df = df[['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            print(f"✅ {timeframe_str}: {len(df):,} bars downloaded")
            print(f"   Date range: {df['Gmt time'].iloc[0]} to {df['Gmt time'].iloc[-1]}")
            print(f"   Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error downloading {timeframe_str}: {e}")
            return None
    
    def download_all_timeframes(self, months_back=36, save_to_disk=True):
        """Download all required timeframes."""
        print(f"{Fore.CYAN}{Style.BRIGHT}🎯 MT5 HISTORICAL DATA DOWNLOADER")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Period: {months_back} months")
        print(f"{'='*60}{Style.RESET_ALL}")
        
        # Timeframe mapping - including 30m as requested
        timeframes = {
            '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1
        }
        
    def download_all_timeframes(self, start_date=None, end_date=None, save_to_disk=True):
        """Download all required timeframes for specified date range."""
        # Default to 2+ years of data as requested
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = datetime(2023, 1, 1)  # Jan 1, 2023 as requested
        
        print(f"{Fore.CYAN}{Style.BRIGHT}🎯 MT5 HISTORICAL DATA DOWNLOADER")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Duration: {(end_date - start_date).days} days ({(end_date - start_date).days/365:.1f} years)")
        print(f"{'='*60}{Style.RESET_ALL}")
        
        # Timeframe mapping - including 30m as requested
        timeframes = {
            '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1
        }
        
        downloaded_data = {}
        
        for tf_name, tf_mt5 in timeframes.items():
            df = self.download_timeframe_data(tf_name, tf_mt5, start_date, end_date)
            
            if df is not None:
                downloaded_data[tf_name] = df
                
                if save_to_disk:
                    # Create data directory if it doesn't exist
                    os.makedirs("data/raw", exist_ok=True)
                    
                    # Save to CSV
                    filename = f"data/raw/XAU_{tf_name}_data.csv"
                    df.to_csv(filename, index=False, sep=';')
                    print(f"💾 Saved: {filename}")
            else:
                downloaded_data[tf_name] = None
        
        return downloaded_data
    
    def get_data_summary(self, data):
        """Print summary of downloaded data."""
        print(f"\n{Fore.GREEN}{Style.BRIGHT}📊 DATA DOWNLOAD SUMMARY")
        print(f"{'='*60}{Style.RESET_ALL}")
        
        total_bars = 0
        date_ranges = []
        
        for tf_name, df in data.items():
            if df is not None:
                bars = len(df)
                total_bars += bars
                start_date = df['Gmt time'].iloc[0]
                end_date = df['Gmt time'].iloc[-1]
                date_ranges.append((start_date, end_date))
                
                print(f"✅ {tf_name.upper()}: {bars:,} bars")
                print(f"   Range: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
                
                # Calculate some basic stats
                price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                volatility = df['Close'].pct_change().std() * 100
                
                print(f"   Price change: {price_change:+.2f}%")
                print(f"   Volatility: {volatility:.2f}%")
                print()
            else:
                print(f"❌ {tf_name.upper()}: No data")
        
        if date_ranges:
            overall_start = min(dr[0] for dr in date_ranges)
            overall_end = max(dr[1] for dr in date_ranges)
            duration = (overall_end - overall_start).days
            
            print(f"{Fore.YELLOW}📈 OVERALL SUMMARY:")
            print(f"   Total bars: {total_bars:,}")
            print(f"   Period: {duration} days ({duration/30:.1f} months)")
            print(f"   Data quality: {'✅ Excellent' if total_bars > 50000 else '⚠️ Limited'}")
        
        print(f"{Style.RESET_ALL}")
    
    def disconnect(self):
        """Disconnect from MT5."""
        if self.connected:
            mt5.shutdown()
            print("✅ Disconnected from MT5")

def main():
    """Main download function."""
    downloader = MT5DataDownloader("XAUUSD")
    
    try:
        # Connect to MT5
        if not downloader.connect():
            return
        
        # Set up date range for 2+ years as requested
        print(f"\n{Fore.YELLOW}📅 DATA DOWNLOAD OPTIONS:")
        print("1. Full range: Jan 1, 2023 - Dec 31, 2025 (3 years)")
        print("2. Recent 2 years: Jan 1, 2024 - Dec 31, 2025")
        print("3. Custom date range")
        
        choice = input(f"\nSelect option (1-3) [1]: {Style.RESET_ALL}").strip() or "1"
        
        if choice == "1":
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2025, 12, 31)
        elif choice == "2":
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2025, 12, 31)
        elif choice == "3":
            start_str = input("Start date (YYYY-MM-DD) [2023-01-01]: ").strip() or "2023-01-01"
            end_str = input("End date (YYYY-MM-DD) [2025-12-31]: ").strip() or "2025-12-31"
            try:
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
            except ValueError:
                print("❌ Invalid date format, using defaults")
                start_date = datetime(2023, 1, 1)
                end_date = datetime(2025, 12, 31)
        else:
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2025, 12, 31)
        
        print(f"\n🚀 Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        print(f"📊 Timeframes: 15m, 30m, 1h, 4h, 1d")
        
        # Download all timeframes
        data = downloader.download_all_timeframes(start_date=start_date, end_date=end_date)
        
        # Show summary
        downloader.get_data_summary(data)
        
        # Verify files were created
        print(f"{Fore.GREEN}📁 Verifying saved files...")
        for tf in ['15m', '30m', '1h', '4h', '1d']:
            filename = f"data/raw/XAU_{tf}_data.csv"
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / (1024 * 1024)
                print(f"✅ {filename} ({size_mb:.1f} MB)")
            else:
                print(f"❌ {filename} not found")
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 DATA DOWNLOAD COMPLETE!")
        print(f"📊 Multi-timeframe data ready for backtesting!")
        print(f"🎯 Use 15m for trade execution, 30m+ for trend analysis")
        print(f"Ready for backtesting with real MT5 data!{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Download cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        downloader.disconnect()

if __name__ == "__main__":
    main()