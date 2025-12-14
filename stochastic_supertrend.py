"""
Stochastic SuperTrend Indicator - Python Implementation
======================================================

Based on the TradingView "Stochastic SuperTrend [BigBeluga]" indicator.
This combines Stochastic RSI with SuperTrend for high-probability signals.

Original Pine Script by BigBeluga
Converted to Python for enhanced trading system.

Key Features:
1. Stochastic RSI calculation
2. SuperTrend applied to Stochastic RSI
3. Dynamic trend detection
4. Buy/Sell signal generation
5. Trend strength measurement
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Tuple, Dict


def calculate_stochastic_rsi(close: pd.Series, 
                           length_rsi: int = 14, 
                           period_k: int = 14, 
                           smooth_k: int = 3) -> pd.Series:
    """
    Calculate Stochastic RSI as used in the original indicator.
    
    Args:
        close: Close price series
        length_rsi: RSI calculation period
        period_k: Stochastic period
        smooth_k: Smoothing period
        
    Returns:
        Stochastic RSI K values
    """
    
    # Calculate RSI
    rsi = ta.rsi(close, length=length_rsi)
    
    # Manual Stochastic RSI calculation (more reliable)
    rsi_min = rsi.rolling(window=period_k).min()
    rsi_max = rsi.rolling(window=period_k).max()
    
    # Avoid division by zero
    rsi_range = rsi_max - rsi_min
    rsi_range = rsi_range.replace(0, np.nan)
    
    stoch_k = 100 * (rsi - rsi_min) / rsi_range
    
    # Apply smoothing (SMA)
    k_smooth = stoch_k.rolling(window=smooth_k).mean()
    
    return k_smooth


def calculate_stochastic_supertrend(close: pd.Series,
                                  length_rsi: int = 14,
                                  period_k: int = 14, 
                                  smooth_k: int = 3,
                                  factor: int = 10) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic SuperTrend indicator.
    
    Args:
        close: Close price series
        length_rsi: RSI calculation period
        period_k: Stochastic period
        smooth_k: Smoothing period
        factor: SuperTrend multiplier
        
    Returns:
        Dictionary containing all indicator components
    """
    
    # Calculate Stochastic RSI
    k = calculate_stochastic_rsi(close, length_rsi, period_k, smooth_k)
    
    # Calculate SuperTrend bands
    upper_band = k + factor
    lower_band = k - factor
    
    # Initialize arrays
    trend = pd.Series(index=k.index, dtype=float)
    direction = pd.Series(index=k.index, dtype=int)
    
    # Calculate SuperTrend logic (vectorized approach)
    prev_upper = upper_band.shift(1).fillna(upper_band)
    prev_lower = lower_band.shift(1).fillna(lower_band)
    prev_k = k.shift(1).fillna(k)
    
    # Adjust bands based on previous values
    upper_band = np.where(
        (upper_band < prev_upper) | (prev_k > prev_upper),
        upper_band,
        prev_upper
    )
    
    lower_band = np.where(
        (lower_band > prev_lower) | (prev_k < prev_lower),
        lower_band,
        prev_lower
    )
    
    # Initialize direction and trend
    direction.iloc[0] = 1
    trend.iloc[0] = lower_band[0] if direction.iloc[0] == -1 else upper_band[0]
    
    # Calculate direction and trend for each bar
    for i in range(1, len(k)):
        if pd.notna(trend.iloc[i-1]):
            if trend.iloc[i-1] == prev_upper[i]:
                direction.iloc[i] = -1 if k.iloc[i] > upper_band[i] else 1
            else:
                direction.iloc[i] = 1 if k.iloc[i] < lower_band[i] else -1
        else:
            direction.iloc[i] = 1
            
        trend.iloc[i] = lower_band[i] if direction.iloc[i] == -1 else upper_band[i]
    
    # Generate signals
    direction_change = direction != direction.shift(1)
    
    # Buy signals: direction changes to -1 (bullish) and k < 50
    buy_signals = direction_change & (direction == -1) & (k < 50)
    
    # Sell signals: direction changes to 1 (bearish) and k > 50  
    sell_signals = direction_change & (direction == 1) & (k > 50)
    
    # Create color coding (following original logic)
    stoch_color = np.where(direction == -1, 1, -1)  # 1 for bullish, -1 for bearish
    
    return {
        'stoch_rsi_k': k,
        'supertrend': trend,
        'direction': direction,
        'buy_signals': buy_signals.astype(int),
        'sell_signals': sell_signals.astype(int),
        'stoch_color': stoch_color,
        'upper_band': pd.Series(upper_band, index=k.index),
        'lower_band': pd.Series(lower_band, index=k.index)
    }


def add_stochastic_supertrend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Stochastic SuperTrend features to dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added Stochastic SuperTrend features
    """
    
    print("Adding Stochastic SuperTrend features...")
    
    # Calculate main indicator
    sst_result = calculate_stochastic_supertrend(
        close=df['Close'],
        length_rsi=14,
        period_k=14,
        smooth_k=3,
        factor=10
    )
    
    # Add all components to dataframe
    df['sst_stoch_rsi'] = sst_result['stoch_rsi_k']
    df['sst_supertrend'] = sst_result['supertrend']
    df['sst_direction'] = sst_result['direction']
    df['sst_buy_signal'] = sst_result['buy_signals']
    df['sst_sell_signal'] = sst_result['sell_signals']
    df['sst_color'] = sst_result['stoch_color']
    df['sst_upper_band'] = sst_result['upper_band']
    df['sst_lower_band'] = sst_result['lower_band']
    
    # Additional derived features for better ML performance
    
    # Trend strength (distance from centerline)
    df['sst_trend_strength'] = abs(df['sst_stoch_rsi'] - 50) / 50
    
    # Momentum (rate of change of Stochastic RSI)
    df['sst_momentum'] = df['sst_stoch_rsi'].diff()
    df['sst_momentum_ma'] = df['sst_momentum'].rolling(5).mean()
    
    # Overbought/Oversold levels
    df['sst_overbought'] = (df['sst_stoch_rsi'] > 80).astype(int)
    df['sst_oversold'] = (df['sst_stoch_rsi'] < 20).astype(int)
    
    # Extreme levels
    df['sst_extreme_high'] = (df['sst_stoch_rsi'] > 90).astype(int)
    df['sst_extreme_low'] = (df['sst_stoch_rsi'] < 10).astype(int)
    
    # Trend persistence (how long current trend has lasted)
    direction_changes = (df['sst_direction'] != df['sst_direction'].shift(1)).astype(int)
    df['sst_trend_age'] = df.groupby(direction_changes.cumsum()).cumcount()
    
    # Signal confirmation (multiple timeframe alignment)
    df['sst_signal_confirmed'] = 0
    
    # Strong bullish: buy signal + oversold + upward momentum
    strong_bull = (df['sst_buy_signal'] == 1) & (df['sst_oversold'] == 1) & (df['sst_momentum'] > 0)
    df.loc[strong_bull, 'sst_signal_confirmed'] = 2
    
    # Weak bullish: buy signal only
    weak_bull = (df['sst_buy_signal'] == 1) & ~strong_bull
    df.loc[weak_bull, 'sst_signal_confirmed'] = 1
    
    # Strong bearish: sell signal + overbought + downward momentum  
    strong_bear = (df['sst_sell_signal'] == 1) & (df['sst_overbought'] == 1) & (df['sst_momentum'] < 0)
    df.loc[strong_bear, 'sst_signal_confirmed'] = -2
    
    # Weak bearish: sell signal only
    weak_bear = (df['sst_sell_signal'] == 1) & ~strong_bear
    df.loc[weak_bear, 'sst_signal_confirmed'] = -1
    
    # Divergence detection (simplified)
    # Price vs Stochastic RSI divergence
    price_direction = np.where(df['Close'] > df['Close'].shift(5), 1, -1)
    sst_direction = np.where(df['sst_stoch_rsi'] > df['sst_stoch_rsi'].shift(5), 1, -1)
    
    df['sst_divergence'] = 0
    df.loc[(price_direction == 1) & (sst_direction == -1), 'sst_divergence'] = -1  # Bearish divergence
    df.loc[(price_direction == -1) & (sst_direction == 1), 'sst_divergence'] = 1   # Bullish divergence
    
    # Trend quality score (0-100)
    # Based on trend age, momentum, and position relative to extremes
    trend_age_score = np.clip(df['sst_trend_age'] / 10, 0, 1)  # Longer trends = higher score
    momentum_score = np.clip(abs(df['sst_momentum']) / 10, 0, 1)  # Stronger momentum = higher score
    position_score = df['sst_trend_strength']  # Distance from center = higher score
    
    df['sst_trend_quality'] = (trend_age_score + momentum_score + position_score) / 3 * 100
    
    print("✓ Stochastic SuperTrend features added")
    
    return df


def get_stochastic_supertrend_signals(df: pd.DataFrame, 
                                    confidence_threshold: float = 0.7) -> pd.DataFrame:
    """
    Generate high-confidence trading signals based on Stochastic SuperTrend.
    
    Args:
        df: DataFrame with Stochastic SuperTrend features
        confidence_threshold: Minimum confidence for signal generation
        
    Returns:
        DataFrame with signal columns added
    """
    
    # High confidence bullish signals
    df['sst_strong_buy'] = (
        (df['sst_buy_signal'] == 1) &
        (df['sst_stoch_rsi'] < 30) &  # Oversold
        (df['sst_momentum'] > 0) &    # Upward momentum
        (df['sst_trend_quality'] > confidence_threshold * 100)
    ).astype(int)
    
    # High confidence bearish signals
    df['sst_strong_sell'] = (
        (df['sst_sell_signal'] == 1) &
        (df['sst_stoch_rsi'] > 70) &  # Overbought
        (df['sst_momentum'] < 0) &    # Downward momentum
        (df['sst_trend_quality'] > confidence_threshold * 100)
    ).astype(int)
    
    # Medium confidence signals
    df['sst_medium_buy'] = (
        (df['sst_buy_signal'] == 1) &
        (df['sst_strong_buy'] == 0) &
        (df['sst_trend_quality'] > (confidence_threshold - 0.2) * 100)
    ).astype(int)
    
    df['sst_medium_sell'] = (
        (df['sst_sell_signal'] == 1) &
        (df['sst_strong_sell'] == 0) &
        (df['sst_trend_quality'] > (confidence_threshold - 0.2) * 100)
    ).astype(int)
    
    # Overall signal strength (0-3)
    df['sst_signal_strength'] = (
        df['sst_strong_buy'] * 3 +
        df['sst_medium_buy'] * 2 +
        df['sst_buy_signal'] * 1 +
        df['sst_strong_sell'] * -3 +
        df['sst_medium_sell'] * -2 +
        df['sst_sell_signal'] * -1
    )
    
    return df


def analyze_stochastic_supertrend_performance(df: pd.DataFrame, 
                                            forward_periods: int = 24) -> Dict:
    """
    Analyze the performance of Stochastic SuperTrend signals.
    
    Args:
        df: DataFrame with signals and price data
        forward_periods: Number of periods to look forward for performance
        
    Returns:
        Performance analysis dictionary
    """
    
    results = {}
    
    # Analyze buy signals
    buy_signals = df[df['sst_buy_signal'] == 1].copy()
    if len(buy_signals) > 0:
        buy_returns = []
        for idx in buy_signals.index:
            if idx + forward_periods < len(df):
                entry_price = df.loc[idx, 'Close']
                exit_price = df.loc[idx + forward_periods, 'Close']
                returns = (exit_price - entry_price) / entry_price
                buy_returns.append(returns)
        
        results['buy_signals'] = {
            'count': len(buy_signals),
            'avg_return': np.mean(buy_returns) if buy_returns else 0,
            'win_rate': len([r for r in buy_returns if r > 0]) / len(buy_returns) if buy_returns else 0,
            'max_return': max(buy_returns) if buy_returns else 0,
            'min_return': min(buy_returns) if buy_returns else 0
        }
    
    # Analyze sell signals
    sell_signals = df[df['sst_sell_signal'] == 1].copy()
    if len(sell_signals) > 0:
        sell_returns = []
        for idx in sell_signals.index:
            if idx + forward_periods < len(df):
                entry_price = df.loc[idx, 'Close']
                exit_price = df.loc[idx + forward_periods, 'Close']
                returns = (entry_price - exit_price) / entry_price  # Short position
                sell_returns.append(returns)
        
        results['sell_signals'] = {
            'count': len(sell_signals),
            'avg_return': np.mean(sell_returns) if sell_returns else 0,
            'win_rate': len([r for r in sell_returns if r > 0]) / len(sell_returns) if sell_returns else 0,
            'max_return': max(sell_returns) if sell_returns else 0,
            'min_return': min(sell_returns) if sell_returns else 0
        }
    
    # Overall performance
    all_returns = buy_returns + sell_returns if 'buy_returns' in locals() and 'sell_returns' in locals() else []
    if all_returns:
        results['overall'] = {
            'total_signals': len(all_returns),
            'avg_return': np.mean(all_returns),
            'win_rate': len([r for r in all_returns if r > 0]) / len(all_returns),
            'sharpe_ratio': np.mean(all_returns) / np.std(all_returns) if np.std(all_returns) > 0 else 0
        }
    
    return results


if __name__ == "__main__":
    # Test the Stochastic SuperTrend indicator
    
    try:
        print("Testing Stochastic SuperTrend Indicator")
        print("=" * 50)
        
        # Load test data
        from enhanced_indicators import load_and_preprocess_data_enhanced
        
        df = load_and_preprocess_data_enhanced(
            "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv",
            normalize=False
        )
        
        print(f"Loaded {len(df)} bars of data")
        
        # Add Stochastic SuperTrend features
        df = add_stochastic_supertrend_features(df)
        df = get_stochastic_supertrend_signals(df)
        
        # Analyze performance
        performance = analyze_stochastic_supertrend_performance(df)
        
        print(f"\nStochastic SuperTrend Analysis:")
        print(f"Total features added: {len([col for col in df.columns if 'sst_' in col])}")
        
        if 'buy_signals' in performance:
            buy_perf = performance['buy_signals']
            print(f"\nBuy Signals Performance:")
            print(f"  Count: {buy_perf['count']}")
            print(f"  Win Rate: {buy_perf['win_rate']*100:.1f}%")
            print(f"  Avg Return: {buy_perf['avg_return']*100:.2f}%")
        
        if 'sell_signals' in performance:
            sell_perf = performance['sell_signals']
            print(f"\nSell Signals Performance:")
            print(f"  Count: {sell_perf['count']}")
            print(f"  Win Rate: {sell_perf['win_rate']*100:.1f}%")
            print(f"  Avg Return: {sell_perf['avg_return']*100:.2f}%")
        
        if 'overall' in performance:
            overall = performance['overall']
            print(f"\nOverall Performance:")
            print(f"  Total Signals: {overall['total_signals']}")
            print(f"  Win Rate: {overall['win_rate']*100:.1f}%")
            print(f"  Avg Return: {overall['avg_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {overall['sharpe_ratio']:.2f}")
        
        # Show signal distribution
        signal_counts = {
            'Buy Signals': df['sst_buy_signal'].sum(),
            'Sell Signals': df['sst_sell_signal'].sum(),
            'Strong Buy': df['sst_strong_buy'].sum(),
            'Strong Sell': df['sst_strong_sell'].sum(),
            'Divergences': abs(df['sst_divergence']).sum()
        }
        
        print(f"\nSignal Distribution:")
        for signal_type, count in signal_counts.items():
            print(f"  {signal_type}: {count}")
        
        # Save sample data
        sample_cols = [col for col in df.columns if 'sst_' in col or col in ['Close', 'Open', 'High', 'Low']]
        df[sample_cols].to_csv("stochastic_supertrend_sample.csv")
        print(f"\nSample data saved to: stochastic_supertrend_sample.csv")
        
        print("\n✓ Stochastic SuperTrend indicator test completed successfully!")
        
    except Exception as e:
        print(f"Error testing Stochastic SuperTrend: {e}")
        import traceback
        traceback.print_exc()