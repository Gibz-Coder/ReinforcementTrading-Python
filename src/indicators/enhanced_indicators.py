"""
Enhanced Technical Indicators with Multi-Timeframe Analysis
===========================================================

Key improvements:
1. Multi-timeframe feature engineering (1H, 4H, 1D)
2. Market session indicators (London, NY, Tokyo)
3. Advanced volatility measures
4. Regime change detection
5. Feature lag analysis for better predictions
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')


def add_market_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market session indicators based on GMT time.
    
    Sessions:
    - Tokyo: 00:00-09:00 GMT
    - London: 08:00-17:00 GMT  
    - New York: 13:00-22:00 GMT
    - Overlaps: London-NY (13:00-17:00), Tokyo-London (08:00-09:00)
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Extract hour from GMT time
    df['hour_gmt'] = df.index.hour
    
    # Session indicators
    df['tokyo_session'] = ((df['hour_gmt'] >= 0) & (df['hour_gmt'] < 9)).astype(int)
    df['london_session'] = ((df['hour_gmt'] >= 8) & (df['hour_gmt'] < 17)).astype(int)
    df['ny_session'] = ((df['hour_gmt'] >= 13) & (df['hour_gmt'] < 22)).astype(int)
    
    # Overlap sessions (high volatility periods)
    df['london_ny_overlap'] = ((df['hour_gmt'] >= 13) & (df['hour_gmt'] < 17)).astype(int)
    df['tokyo_london_overlap'] = ((df['hour_gmt'] >= 8) & (df['hour_gmt'] < 9)).astype(int)
    
    # Session activity score (0-3 based on active sessions)
    df['session_activity'] = (df['tokyo_session'] + df['london_session'] + df['ny_session'])
    
    return df


def create_multi_timeframe_features(df: pd.DataFrame, base_timeframe='1H') -> pd.DataFrame:
    """
    Create multi-timeframe features by resampling to higher timeframes.
    
    Args:
        df: Hourly OHLCV data
        base_timeframe: Base timeframe (default '1H')
    
    Returns:
        DataFrame with multi-timeframe features
    """
    # Ensure proper datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Create 4H features
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max', 
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Create daily features  
    df_daily = df.resample('1D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min', 
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Calculate 4H indicators
    df_4h['ma_20_4h'] = ta.sma(df_4h['Close'], length=20)
    df_4h['ma_50_4h'] = ta.sma(df_4h['Close'], length=50)
    df_4h['rsi_14_4h'] = ta.rsi(df_4h['Close'], length=14)
    df_4h['atr_14_4h'] = ta.atr(df_4h['High'], df_4h['Low'], df_4h['Close'], length=14)
    
    # MACD for 4H
    macd_4h = ta.macd(df_4h['Close'], fast=12, slow=26, signal=9)
    if macd_4h is not None:
        df_4h['macd_4h'] = macd_4h['MACD_12_26_9']
        df_4h['macd_signal_4h'] = macd_4h['MACDs_12_26_9']
    
    # Calculate daily indicators
    df_daily['ma_20_daily'] = ta.sma(df_daily['Close'], length=20)
    df_daily['ma_50_daily'] = ta.sma(df_daily['Close'], length=50)
    df_daily['rsi_14_daily'] = ta.rsi(df_daily['Close'], length=14)
    df_daily['atr_14_daily'] = ta.atr(df_daily['High'], df_daily['Low'], df_daily['Close'], length=14)
    
    # Daily trend strength
    df_daily['daily_trend'] = np.where(df_daily['Close'] > df_daily['ma_20_daily'], 1,
                                      np.where(df_daily['Close'] < df_daily['ma_20_daily'], -1, 0))
    
    # Merge higher timeframe data back to hourly
    # Forward fill to propagate values to all hours
    df = df.merge(df_4h[['ma_20_4h', 'ma_50_4h', 'rsi_14_4h', 'atr_14_4h', 'macd_4h', 'macd_signal_4h']], 
                  left_index=True, right_index=True, how='left')
    df = df.merge(df_daily[['ma_20_daily', 'ma_50_daily', 'rsi_14_daily', 'atr_14_daily', 'daily_trend']], 
                  left_index=True, right_index=True, how='left')
    
    # Forward fill missing values
    mtf_cols = ['ma_20_4h', 'ma_50_4h', 'rsi_14_4h', 'atr_14_4h', 'macd_4h', 'macd_signal_4h',
                'ma_20_daily', 'ma_50_daily', 'rsi_14_daily', 'atr_14_daily', 'daily_trend']
    
    for col in mtf_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill')
    
    # Multi-timeframe alignment signals
    df['mtf_trend_alignment'] = 0
    
    # All timeframes bullish
    bullish_1h = df['Close'] > df.get('ma_20', df['Close'])
    bullish_4h = df['Close'] > df.get('ma_20_4h', df['Close']) 
    bullish_daily = df.get('daily_trend', 0) == 1
    
    df.loc[bullish_1h & bullish_4h & bullish_daily, 'mtf_trend_alignment'] = 2  # Strong bull
    df.loc[bullish_1h & bullish_4h & ~bullish_daily, 'mtf_trend_alignment'] = 1  # Medium bull
    
    # All timeframes bearish
    bearish_1h = df['Close'] < df.get('ma_20', df['Close'])
    bearish_4h = df['Close'] < df.get('ma_20_4h', df['Close'])
    bearish_daily = df.get('daily_trend', 0) == -1
    
    df.loc[bearish_1h & bearish_4h & bearish_daily, 'mtf_trend_alignment'] = -2  # Strong bear
    df.loc[bearish_1h & bearish_4h & ~bearish_daily, 'mtf_trend_alignment'] = -1  # Medium bear
    
    return df


def add_advanced_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced volatility and regime detection features."""
    
    # Realized volatility (rolling standard deviation of returns)
    df['returns'] = df['Close'].pct_change()
    df['realized_vol_24h'] = df['returns'].rolling(24).std() * np.sqrt(24)  # 24-hour vol
    df['realized_vol_168h'] = df['returns'].rolling(168).std() * np.sqrt(168)  # Weekly vol
    
    # Volatility regime (high/low vol periods)
    vol_median = df['realized_vol_24h'].rolling(720).median()  # 30-day median
    df['vol_regime'] = np.where(df['realized_vol_24h'] > vol_median * 1.5, 1,  # High vol
                               np.where(df['realized_vol_24h'] < vol_median * 0.5, -1, 0))  # Low vol
    
    # Parkinson volatility (uses High-Low range)
    df['parkinson_vol'] = np.sqrt(0.361 * (np.log(df['High'] / df['Low']) ** 2).rolling(24).mean())
    
    # Garman-Klass volatility (more efficient estimator)
    hl_term = 0.5 * (np.log(df['High'] / df['Low']) ** 2)
    co_term = (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
    df['gk_volatility'] = np.sqrt((hl_term - co_term).rolling(24).mean())
    
    # Volume-weighted volatility
    df['vwap'] = (df['Close'] * df['Volume']).rolling(24).sum() / df['Volume'].rolling(24).sum()
    df['vwap_deviation'] = abs(df['Close'] - df['vwap']) / df['vwap']
    
    return df


def add_regime_change_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime change detection using multiple methods.
    """
    
    # Price momentum regime changes
    df['momentum_20'] = df['Close'].pct_change(20)
    df['momentum_regime'] = np.where(df['momentum_20'] > 0.02, 1,  # Strong uptrend
                                    np.where(df['momentum_20'] < -0.02, -1, 0))  # Strong downtrend
    
    # Volatility breakouts
    df['vol_breakout'] = 0
    vol_ma = df['realized_vol_24h'].rolling(168).mean()  # Weekly average
    vol_std = df['realized_vol_24h'].rolling(168).std()
    
    # High volatility breakout
    df.loc[df['realized_vol_24h'] > vol_ma + 2 * vol_std, 'vol_breakout'] = 1
    # Low volatility compression
    df.loc[df['realized_vol_24h'] < vol_ma - vol_std, 'vol_breakout'] = -1
    
    # Trend strength changes (ADX-based)
    if 'adx' in df.columns:
        df['adx_change'] = df['adx'].diff(5)  # 5-period ADX change
        df['trend_strength_change'] = np.where(df['adx_change'] > 0.1, 1,  # Strengthening
                                              np.where(df['adx_change'] < -0.1, -1, 0))  # Weakening
    
    # Support/Resistance levels (using rolling min/max)
    df['resistance_20'] = df['High'].rolling(20).max()
    df['support_20'] = df['Low'].rolling(20).min()
    df['near_resistance'] = (df['Close'] / df['resistance_20'] > 0.995).astype(int)
    df['near_support'] = (df['Close'] / df['support_20'] < 1.005).astype(int)
    
    return df


def add_lag_features(df: pd.DataFrame, feature_cols: list, lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """
    Add lagged features for better prediction capability.
    
    Args:
        df: DataFrame with features
        feature_cols: List of columns to create lags for
        lags: List of lag periods
    """
    
    for col in feature_cols:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df


def load_and_preprocess_data_enhanced(csv_path: str, normalize: bool = True) -> pd.DataFrame:
    """
    Enhanced version of data preprocessing with multi-timeframe analysis.
    
    Key improvements:
    1. Multi-timeframe features (1H, 4H, Daily)
    2. Market session indicators
    3. Advanced volatility measures
    4. Regime change detection
    5. Lag features for prediction
    """
    
    print(f"Loading enhanced data from: {csv_path}")
    
    # Load base data with proper date parsing
    df = pd.read_csv(csv_path)
    
    # Convert the date column with the correct format (DD.MM.YYYY HH:MM:SS.fff)
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
    df.set_index('Gmt time', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Base data loaded: {len(df)} rows")
    
    # Add market session indicators
    df = add_market_sessions(df)
    print("✓ Market sessions added")
    
    # Add basic technical indicators (from original function)
    # Moving Averages
    df['ma_20'] = ta.sma(df['Close'], length=20)
    df['ma_50'] = ta.sma(df['Close'], length=50)
    df['ma_200'] = ta.sma(df['Close'], length=200)
    df['ema_12'] = ta.ema(df['Close'], length=12)
    df['ema_26'] = ta.ema(df['Close'], length=26)
    
    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
    
    # RSI
    df['rsi_14'] = ta.rsi(df['Close'], length=14)
    
    # ATR
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # ADX
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx is not None:
        df['adx'] = adx['ADX_14']
        df['plus_di'] = adx['DMP_14']
        df['minus_di'] = adx['DMN_14']
    
    # Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None:
        bb_cols = bbands.columns.tolist()
        bb_upper_col = [c for c in bb_cols if c.startswith('BBU')][0]
        bb_middle_col = [c for c in bb_cols if c.startswith('BBM')][0]
        bb_lower_col = [c for c in bb_cols if c.startswith('BBL')][0]
        
        df['bb_upper'] = bbands[bb_upper_col]
        df['bb_middle'] = bbands[bb_middle_col]
        df['bb_lower'] = bbands[bb_lower_col]
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    print("✓ Basic technical indicators added")
    
    # Add multi-timeframe features
    df = create_multi_timeframe_features(df)
    print("✓ Multi-timeframe features added")
    
    # Add advanced volatility features
    df = add_advanced_volatility_features(df)
    print("✓ Advanced volatility features added")
    
    # Add regime change detection
    df = add_regime_change_detection(df)
    print("✓ Regime change detection added")
    
    # Add lag features for key indicators
    key_features = ['rsi_14', 'macd', 'atr', 'adx', 'bb_pct', 'realized_vol_24h']
    df = add_lag_features(df, key_features, lags=[1, 2, 3, 5])
    print("✓ Lag features added")
    
    # Market regime classification (enhanced)
    df['market_regime'] = 0
    
    # Strong uptrend: Multiple timeframes aligned + low volatility
    strong_bull = ((df['mtf_trend_alignment'] >= 1) & 
                   (df['adx'] > 25) & 
                   (df['plus_di'] > df['minus_di']) &
                   (df['vol_regime'] <= 0))
    df.loc[strong_bull, 'market_regime'] = 2
    
    # Moderate uptrend
    mod_bull = ((df['Close'] > df['ma_20']) & 
                (df['rsi_14'] > 50) & 
                (df['macd'] > df['macd_signal']))
    df.loc[mod_bull & ~strong_bull, 'market_regime'] = 1
    
    # Strong downtrend
    strong_bear = ((df['mtf_trend_alignment'] <= -1) & 
                   (df['adx'] > 25) & 
                   (df['minus_di'] > df['plus_di']) &
                   (df['vol_regime'] <= 0))
    df.loc[strong_bear, 'market_regime'] = -2
    
    # Moderate downtrend
    mod_bear = ((df['Close'] < df['ma_20']) & 
                (df['rsi_14'] < 50) & 
                (df['macd'] < df['macd_signal']))
    df.loc[mod_bear & ~strong_bear, 'market_regime'] = -1
    
    print("✓ Enhanced market regime classification added")
    
    # Clean data
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    print(f"Data after cleaning: {len(df)} rows, {df.shape[1]} features")
    
    # Normalization
    if normalize:
        print("Normalizing features...")
        
        # Price normalization (percentage change from first value)
        first_close = df['Close'].iloc[0]
        price_cols = ['Open', 'High', 'Low', 'Close', 'ma_20', 'ma_50', 'ma_200', 
                      'ema_12', 'ema_26', 'bb_upper', 'bb_middle', 'bb_lower',
                      'ma_20_4h', 'ma_50_4h', 'ma_20_daily', 'ma_50_daily']
        
        for col in price_cols:
            if col in df.columns:
                df[col] = (df[col] / first_close - 1) * 100
        
        # Normalize oscillators to [-1, 1] range
        oscillator_cols = ['rsi_14', 'rsi_14_4h', 'rsi_14_daily']
        for col in oscillator_cols:
            if col in df.columns:
                df[col] = (df[col] - 50) / 50
        
        # Normalize ADX to [0, 1] range
        adx_cols = ['adx', 'plus_di', 'minus_di']
        for col in adx_cols:
            if col in df.columns:
                df[col] = df[col] / 50.0
        
        # Normalize volatility measures using z-score
        vol_cols = ['atr', 'atr_14_4h', 'atr_14_daily', 'realized_vol_24h', 
                    'realized_vol_168h', 'parkinson_vol', 'gk_volatility']
        for col in vol_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[col] = (df[col] - mean_val) / (std_val + 1e-10)
        
        # Clip extreme values
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].clip(-10, 10)
        
        print("✓ Normalization completed")
    
    # Final cleanup
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    print(f"Final dataset: {len(df)} rows, {df.shape[1]} features")
    print("Enhanced preprocessing completed!")
    
    return df


def get_enhanced_feature_names() -> list:
    """Returns comprehensive list of all enhanced features."""
    
    base_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'hour_gmt', 'tokyo_session', 'london_session', 'ny_session',
        'london_ny_overlap', 'tokyo_london_overlap', 'session_activity'
    ]
    
    technical_features = [
        'ma_20', 'ma_50', 'ma_200', 'ema_12', 'ema_26',
        'macd', 'macd_signal', 'macd_hist', 'rsi_14', 'atr',
        'adx', 'plus_di', 'minus_di', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct'
    ]
    
    mtf_features = [
        'ma_20_4h', 'ma_50_4h', 'rsi_14_4h', 'atr_14_4h', 'macd_4h', 'macd_signal_4h',
        'ma_20_daily', 'ma_50_daily', 'rsi_14_daily', 'atr_14_daily', 'daily_trend',
        'mtf_trend_alignment'
    ]
    
    volatility_features = [
        'returns', 'realized_vol_24h', 'realized_vol_168h', 'vol_regime',
        'parkinson_vol', 'gk_volatility', 'vwap', 'vwap_deviation'
    ]
    
    regime_features = [
        'momentum_20', 'momentum_regime', 'vol_breakout', 'trend_strength_change',
        'resistance_20', 'support_20', 'near_resistance', 'near_support', 'market_regime'
    ]
    
    # Lag features (examples - actual list would be longer)
    lag_features = [
        'rsi_14_lag_1', 'rsi_14_lag_2', 'rsi_14_lag_3', 'rsi_14_lag_5',
        'macd_lag_1', 'macd_lag_2', 'macd_lag_3', 'macd_lag_5',
        'atr_lag_1', 'atr_lag_2', 'atr_lag_3', 'atr_lag_5'
    ]
    
    return base_features + technical_features + mtf_features + volatility_features + regime_features + lag_features


if __name__ == "__main__":
    # Test the enhanced preprocessing
    test_file = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    
    try:
        df = load_and_preprocess_data_enhanced(test_file, normalize=True)
        print(f"\nTest successful! Dataset shape: {df.shape}")
        print(f"Feature columns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df.head())
        
        # Save enhanced dataset for inspection
        df.to_csv("enhanced_data_sample.csv")
        print(f"\nSample saved to: enhanced_data_sample.csv")
        
    except FileNotFoundError:
        print(f"Test file not found: {test_file}")
        print("Please ensure the data file exists to test the enhanced preprocessing.")