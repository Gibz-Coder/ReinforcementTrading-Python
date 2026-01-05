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
    
    def download_timeframe_data(self, timeframe_str, timeframe_mt5, months_back=12):
        """Download data for specific timeframe."""
        if not self.connected:
            print("❌ Not connected to MT5")
            return None
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)  # Approximate
        
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
        
        # Timeframe mapping
        timeframes = {
            '15m': mt5.TIMEFRAME_M15,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1
        }
        
        downloaded_data = {}
        
        for tf_name, tf_mt5 in timeframes.items():
            df = self.download_timeframe_data(tf_name, tf_mt5, months_back)
            
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
        
        # Download data (12 months by default)
        print(f"\n{Fore.YELLOW}How many months of data to download?")
        print("1. 12 months (quick)")
        print("2. 24 months (recommended)")
        print("3. 36 months (comprehensive)")
        print("4. 48 months (maximum)")
        
        choice = input(f"\nSelect option (1-4) [3]: {Style.RESET_ALL}").strip()
        
        months_map = {'1': 12, '2': 24, '3': 36, '4': 48}
        months = months_map.get(choice, 36)  # Default to 36 months
        
        print(f"\n🚀 Downloading {months} months of data...")
        
        # Download all timeframes
        data = downloader.download_all_timeframes(months_back=months)
        
        # Show summary
        downloader.get_data_summary(data)
        
        # Verify files were created
        print(f"{Fore.GREEN}📁 Verifying saved files...")
        for tf in ['15m', '1h', '4h', '1d']:
            filename = f"data/raw/XAU_{tf}_data.csv"
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / (1024 * 1024)
                print(f"✅ {filename} ({size_mb:.1f} MB)")
            else:
                print(f"❌ {filename} not found")
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 DATA DOWNLOAD COMPLETE!")
        print(f"Ready for backtesting with fresh MT5 data!{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Download cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        downloader.disconnect()

if __name__ == "__main__":
    main()