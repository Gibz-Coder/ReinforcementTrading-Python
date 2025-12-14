"""
High Win Rate Indicators - Advanced Feature Engineering
======================================================

Focus on features that have proven predictive power for high win rates:
1. Price action patterns
2. Volume-price analysis
3. Market microstructure indicators
4. Momentum divergence detection
5. Support/resistance strength
6. Volatility regime transitions
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def add_price_action_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price action pattern recognition for higher win rates.
    
    These patterns have historically shown good predictive power.
    """
    
    # Candlestick patterns
    df['doji'] = (abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10) < 0.1).astype(int)
    df['hammer'] = ((df['Close'] > df['Open']) & 
                    ((df['Open'] - df['Low']) > 2 * (df['Close'] - df['Open'])) &
                    ((df['High'] - df['Close']) < 0.3 * (df['Close'] - df['Open']))).astype(int)
    
    df['shooting_star'] = ((df['Open'] > df['Close']) & 
                          ((df['High'] - df['Open']) > 2 * (df['Open'] - df['Close'])) &
                          ((df['Close'] - df['Low']) < 0.3 * (df['Open'] - df['Close']))).astype(int)
    
    # Engulfing patterns
    df['bullish_engulfing'] = ((df['Close'] > df['Open']) & 
                              (df['Close'].shift(1) < df['Open'].shift(1)) &
                              (df['Open'] < df['Close'].shift(1)) &
                              (df['Close'] > df['Open'].shift(1))).astype(int)
    
    df['bearish_engulfing'] = ((df['Close'] < df['Open']) & 
                              (df['Close'].shift(1) > df['Open'].shift(1)) &
                              (df['Open'] > df['Close'].shift(1)) &
                              (df['Close'] < df['Open'].shift(1))).astype(int)
    
    # Inside/Outside bars
    df['inside_bar'] = ((df['High'] < df['High'].shift(1)) & 
                       (df['Low'] > df['Low'].shift(1))).astype(int)
    
    df['outside_bar'] = ((df['High'] > df['High'].shift(1)) & 
                        (df['Low'] < df['Low'].shift(1))).astype(int)
    
    # Price gaps
    df['gap_up'] = (df['Low'] > df['High'].shift(1)).astype(int)
    df['gap_down'] = (df['High'] < df['Low'].shift(1)).astype(int)
    
    return df


def add_volume_price_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-price analysis indicators for better entry timing.
    """
    
    # Volume moving averages
    df['volume_ma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ma_50'] = df['Volume'].rolling(50).mean()
    
    # Volume relative to average
    df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
    df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
    df['low_volume'] = (df['volume_ratio'] < 0.5).astype(int)
    
    # Price-Volume Trend (PVT)
    df['price_change_pct'] = df['Close'].pct_change()
    df['pvt'] = (df['price_change_pct'] * df['Volume']).cumsum()
    df['pvt_ma'] = df['pvt'].rolling(20).mean()
    df['pvt_signal'] = (df['pvt'] > df['pvt_ma']).astype(int)
    
    # Volume-Weighted Average Price (VWAP) deviation
    df['vwap'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    df['vwap_deviation'] = (df['Close'] - df['vwap']) / df['vwap']
    df['above_vwap'] = (df['Close'] > df['vwap']).astype(int)
    
    # On-Balance Volume (OBV)
    df['obv'] = ta.obv(df['Close'], df['Volume'])
    df['obv_ma'] = df['obv'].rolling(20).mean()
    df['obv_signal'] = (df['obv'] > df['obv_ma']).astype(int)
    
    return df


def add_momentum_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum divergence detection for high-probability setups.
    """
    
    # RSI divergence
    df['rsi'] = ta.rsi(df['Close'], length=14)
    
    # Price peaks and troughs
    df['price_peak'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    df['price_trough'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    
    # RSI peaks and troughs
    df['rsi_peak'] = (df['rsi'] > df['rsi'].shift(1)) & (df['rsi'] > df['rsi'].shift(-1))
    df['rsi_trough'] = (df['rsi'] < df['rsi'].shift(1)) & (df['rsi'] < df['rsi'].shift(-1))
    
    # Bullish divergence: price makes lower low, RSI makes higher low
    df['bullish_divergence'] = 0
    df['bearish_divergence'] = 0
    
    # MACD divergence
    macd_data = ta.macd(df['Close'])
    if macd_data is not None:
        df['macd'] = macd_data['MACD_12_26_9']
        df['macd_signal'] = macd_data['MACDs_12_26_9']
        df['macd_histogram'] = macd_data['MACDh_12_26_9']
        
        # MACD crossover signals
        df['macd_bullish_cross'] = ((df['macd'] > df['macd_signal']) & 
                                   (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_bearish_cross'] = ((df['macd'] < df['macd_signal']) & 
                                   (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
    
    return df


def add_support_resistance_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dynamic support/resistance levels with strength indicators.
    """
    
    # Pivot points
    df['pivot_high'] = ((df['High'] > df['High'].shift(1)) & 
                       (df['High'] > df['High'].shift(-1)) &
                       (df['High'] > df['High'].shift(2)) & 
                       (df['High'] > df['High'].shift(-2))).astype(int)
    
    df['pivot_low'] = ((df['Low'] < df['Low'].shift(1)) & 
                      (df['Low'] < df['Low'].shift(-1)) &
                      (df['Low'] < df['Low'].shift(2)) & 
                      (df['Low'] < df['Low'].shift(-2))).astype(int)
    
    # Dynamic support/resistance levels
    window = 50
    df['resistance_level'] = df['High'].rolling(window).max()
    df['support_level'] = df['Low'].rolling(window).min()
    
    # Distance to support/resistance
    df['dist_to_resistance'] = (df['resistance_level'] - df['Close']) / df['Close']
    df['dist_to_support'] = (df['Close'] - df['support_level']) / df['Close']
    
    # Near support/resistance (within 0.1%)
    df['near_resistance'] = (df['dist_to_resistance'] < 0.001).astype(int)
    df['near_support'] = (df['dist_to_support'] < 0.001).astype(int)
    
    # Breakout signals
    df['resistance_break'] = ((df['Close'] > df['resistance_level'].shift(1)) & 
                             (df['Close'].shift(1) <= df['resistance_level'].shift(1))).astype(int)
    df['support_break'] = ((df['Close'] < df['support_level'].shift(1)) & 
                          (df['Close'].shift(1) >= df['support_level'].shift(1))).astype(int)
    
    return df


def add_volatility_regime_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility regime transition indicators for better timing.
    """
    
    # True Range and ATR
    df['tr'] = ta.true_range(df['High'], df['Low'], df['Close'])
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['atr_ma'] = df['atr'].rolling(50).mean()
    
    # Volatility regimes
    df['vol_expansion'] = (df['atr'] > df['atr_ma'] * 1.5).astype(int)
    df['vol_contraction'] = (df['atr'] < df['atr_ma'] * 0.5).astype(int)
    
    # Volatility breakout
    df['vol_breakout'] = ((df['atr'] > df['atr_ma'] * 1.5) & 
                         (df['atr'].shift(1) <= df['atr_ma'].shift(1) * 1.5)).astype(int)
    
    # Bollinger Band squeeze
    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None:
        bb_cols = bbands.columns.tolist()
        bb_upper_col = [c for c in bb_cols if c.startswith('BBU')][0]
        bb_lower_col = [c for c in bb_cols if c.startswith('BBL')][0]
        
        df['bb_upper'] = bbands[bb_upper_col]
        df['bb_lower'] = bbands[bb_lower_col]
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['Close']
        df['bb_width_ma'] = df['bb_width'].rolling(20).mean()
        
        # Squeeze detection
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width_ma'] * 0.8).astype(int)
        df['bb_expansion'] = (df['bb_width'] > df['bb_width_ma'] * 1.2).astype(int)
    
    return df


def add_market_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market microstructure indicators for better execution.
    """
    
    # Bid-Ask spread proxy (using high-low range)
    df['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
    df['spread_ma'] = df['spread_proxy'].rolling(20).mean()
    df['tight_spread'] = (df['spread_proxy'] < df['spread_ma'] * 0.8).astype(int)
    df['wide_spread'] = (df['spread_proxy'] > df['spread_ma'] * 1.2).astype(int)
    
    # Price efficiency (how much price moves relative to range)
    df['price_efficiency'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
    df['efficient_move'] = (df['price_efficiency'] > 0.7).astype(int)
    
    # Momentum persistence
    df['momentum_1h'] = df['Close'].pct_change(1)
    df['momentum_4h'] = df['Close'].pct_change(4)
    df['momentum_24h'] = df['Close'].pct_change(24)
    
    # Momentum alignment
    df['momentum_aligned_bull'] = ((df['momentum_1h'] > 0) & 
                                  (df['momentum_4h'] > 0) & 
                                  (df['momentum_24h'] > 0)).astype(int)
    df['momentum_aligned_bear'] = ((df['momentum_1h'] < 0) & 
                                  (df['momentum_4h'] < 0) & 
                                  (df['momentum_24h'] < 0)).astype(int)
    
    return df


def add_machine_learning_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add machine learning derived features for pattern recognition.
    """
    
    # Rolling correlations between price and volume
    df['price_volume_corr'] = df['Close'].rolling(20).corr(df['Volume'])
    
    # Price acceleration (second derivative)
    df['price_velocity'] = df['Close'].diff()
    df['price_acceleration'] = df['price_velocity'].diff()
    
    # Fractal dimension (complexity measure)
    def fractal_dimension(series, window=20):
        """Calculate fractal dimension of price series."""
        result = []
        for i in range(len(series)):
            if i < window:
                result.append(np.nan)
            else:
                data = series.iloc[i-window:i].values
                # Simplified fractal dimension calculation
                n = len(data)
                if n > 1:
                    # Calculate Hurst exponent approximation
                    lags = range(2, min(n//2, 20))
                    tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    hurst = poly[0] * 2.0
                    fractal_dim = 2 - hurst
                    result.append(fractal_dim)
                else:
                    result.append(np.nan)
        return pd.Series(result, index=series.index)
    
    df['fractal_dimension'] = fractal_dimension(df['Close'])
    
    # Regime stability (how long current regime has persisted)
    df['returns'] = df['Close'].pct_change()
    df['regime'] = np.where(df['returns'] > 0, 1, -1)
    df['regime_changes'] = (df['regime'] != df['regime'].shift(1)).astype(int)
    df['regime_stability'] = df.groupby((df['regime_changes'] == 1).cumsum())['regime'].cumcount()
    
    return df


def create_high_winrate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive feature set optimized for high win rates.
    """
    
    print("Creating high win rate features...")
    
    # Add all feature categories
    df = add_price_action_patterns(df)
    print("✓ Price action patterns added")
    
    df = add_volume_price_analysis(df)
    print("✓ Volume-price analysis added")
    
    df = add_momentum_divergence(df)
    print("✓ Momentum divergence indicators added")
    
    df = add_support_resistance_strength(df)
    print("✓ Support/resistance strength added")
    
    df = add_volatility_regime_transitions(df)
    print("✓ Volatility regime transitions added")
    
    df = add_market_microstructure(df)
    print("✓ Market microstructure indicators added")
    
    df = add_machine_learning_features(df)
    print("✓ Machine learning features added")
    
    return df


def create_predictive_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create high-probability trading signals based on feature combinations.
    """
    
    # High-probability bullish signals
    df['strong_bull_signal'] = (
        (df['bullish_engulfing'] == 1) |
        (df['hammer'] == 1) |
        (df['macd_bullish_cross'] == 1) |
        (df['resistance_break'] == 1) |
        (df['momentum_aligned_bull'] == 1)
    ).astype(int)
    
    # High-probability bearish signals
    df['strong_bear_signal'] = (
        (df['bearish_engulfing'] == 1) |
        (df['shooting_star'] == 1) |
        (df['macd_bearish_cross'] == 1) |
        (df['support_break'] == 1) |
        (df['momentum_aligned_bear'] == 1)
    ).astype(int)
    
    # Signal strength (number of confirming indicators)
    bull_indicators = ['bullish_engulfing', 'hammer', 'macd_bullish_cross', 'resistance_break', 
                      'momentum_aligned_bull', 'high_volume', 'vol_breakout', 'obv_signal']
    bear_indicators = ['bearish_engulfing', 'shooting_star', 'macd_bearish_cross', 'support_break',
                      'momentum_aligned_bear', 'high_volume', 'vol_breakout']
    
    df['bull_signal_strength'] = df[bull_indicators].sum(axis=1)
    df['bear_signal_strength'] = df[bear_indicators].sum(axis=1)
    
    # High-confidence signals (3+ confirming indicators)
    df['high_confidence_bull'] = (df['bull_signal_strength'] >= 3).astype(int)
    df['high_confidence_bear'] = (df['bear_signal_strength'] >= 3).astype(int)
    
    return df


if __name__ == "__main__":
    # Test the high win rate indicators
    from enhanced_indicators import load_and_preprocess_data_enhanced
    
    try:
        print("Testing high win rate indicators...")
        
        # Load base data
        df = load_and_preprocess_data_enhanced(
            "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv",
            normalize=False  # Don't normalize yet
        )
        
        # Add high win rate features
        df = create_high_winrate_features(df)
        df = create_predictive_signals(df)
        
        print(f"\nHigh win rate features created successfully!")
        print(f"Total features: {df.shape[1]}")
        print(f"Signal distribution:")
        print(f"  Strong bull signals: {df['strong_bull_signal'].sum()}")
        print(f"  Strong bear signals: {df['strong_bear_signal'].sum()}")
        print(f"  High confidence bull: {df['high_confidence_bull'].sum()}")
        print(f"  High confidence bear: {df['high_confidence_bear'].sum()}")
        
        # Save sample
        df.to_csv("high_winrate_features_sample.csv")
        print(f"Sample saved to: high_winrate_features_sample.csv")
        
    except Exception as e:
        print(f"Error testing high win rate indicators: {e}")
        import traceback
        traceback.print_exc()