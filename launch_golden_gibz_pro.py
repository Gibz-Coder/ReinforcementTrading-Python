#!/usr/bin/env python3
"""
Golden Gibz Professional EA Launcher
===================================
Easy launcher for the enhanced Golden Gibz EA
"""

import os
import sys
from colorama import init, Fore, Style

init(autoreset=True)

def main():
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*80}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}🎯 GOLDEN GIBZ PROFESSIONAL EA LAUNCHER")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}")
    print(f"{Fore.GREEN}🚀 Enhanced Features:")
    print(f"   • Beautiful Real-time Dashboard")
    print(f"   • Advanced Risk Management")
    print(f"   • Configurable Trading Hours")
    print(f"   • Dynamic Position Sizing")
    print(f"   • Technical Indicators Display")
    print(f"   • Interactive Configuration Menu")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
    
    print(f"\n{Fore.YELLOW}Choose launch option:{Style.RESET_ALL}")
    print(f"1. {Fore.GREEN}Quick Start{Style.RESET_ALL} (Use existing config)")
    print(f"2. {Fore.BLUE}Configure Settings{Style.RESET_ALL} (Interactive setup)")
    print(f"3. {Fore.MAGENTA}View Current Config{Style.RESET_ALL}")
    print(f"4. {Fore.RED}Exit{Style.RESET_ALL}")
    
    choice = input(f"\n{Fore.YELLOW}Select option (1-4): {Style.RESET_ALL}")
    
    if choice == '1':
        print(f"\n{Fore.GREEN}🚀 Starting Golden Gibz Professional EA...{Style.RESET_ALL}")
        os.system('python golden_gibz_python_ea.py')
    elif choice == '2':
        print(f"\n{Fore.BLUE}⚙️ Starting with configuration menu...{Style.RESET_ALL}")
        # Set environment variable to force config menu
        os.environ['FORCE_CONFIG'] = '1'
        os.system('python golden_gibz_python_ea.py')
    elif choice == '3':
        show_current_config()
    elif choice == '4':
        print(f"{Fore.RED}👋 Goodbye!{Style.RESET_ALL}")
        sys.exit(0)
    else:
        print(f"{Fore.RED}❌ Invalid option{Style.RESET_ALL}")
        main()

def show_current_config():
    """Display current configuration."""
    try:
        import json
        
        print(f"\n{Fore.CYAN}{Style.BRIGHT}📋 CURRENT CONFIGURATION{Style.RESET_ALL}")
        print(f"{'='*50}")
        
        with open('config/ea_config.json', 'r') as f:
            config = json.load(f)
        
        print(f"{Fore.GREEN}Trading Parameters:{Style.RESET_ALL}")
        print(f"  Symbol: {config['symbol']}")
        print(f"  Lot Size: {config['lot_size']}")
        print(f"  Max Positions: {config['max_positions']}")
        print(f"  Min Confidence: {config['min_confidence']}")
        print(f"  Signal Frequency: {config['signal_frequency']}s")
        
        print(f"\n{Fore.RED}Risk Management:{Style.RESET_ALL}")
        print(f"  Max Daily Trades: {config['max_daily_trades']}")
        print(f"  Max Daily Loss: ${config['max_daily_loss']}")
        print(f"  Risk per Trade: {config['risk_per_trade']}%")
        print(f"  Dynamic Lots: {config['use_dynamic_lots']}")
        
        print(f"\n{Fore.YELLOW}Trading Hours:{Style.RESET_ALL}")
        print(f"  Start: {config['trading_hours']['start']:02d}:00")
        print(f"  End: {config['trading_hours']['end']:02d}:00")
        
        print(f"\n{Fore.MAGENTA}Dashboard:{Style.RESET_ALL}")
        print(f"  Refresh Rate: {config['dashboard_refresh']}s")
        print(f"  Show Indicators: {config['show_indicators']}")
        print(f"  Show Positions: {config['show_positions']}")
        
        input(f"\n{Fore.CYAN}Press Enter to return to menu...{Style.RESET_ALL}")
        main()
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error reading config: {e}{Style.RESET_ALL}")
        input(f"{Fore.CYAN}Press Enter to return to menu...{Style.RESET_ALL}")
        main()

if __name__ == "__main__":
    main()