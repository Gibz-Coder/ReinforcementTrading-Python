import pandas as pd
import pandas_ta as ta
import numpy as np

def load_and_preprocess_data(csv_path: str, normalize: bool = True) -> pd.DataFrame:
    """
    Loads EURUSD data from CSV and preprocesses it by adding technical indicators.
    Expects columns: [Gmt time, Open, High, Low, Close, Volume].

    Enhanced with:
    - MACD (trend momentum)
    - Bollinger Bands (volatility)
    - Stochastic Oscillator (momentum)
    - Price rate of change
    - Normalized features for better RL training
    """
    df = pd.read_csv(csv_path, parse_dates=True, index_col='Gmt time')

    # Sort by date just in case
    df.sort_index(inplace=True)

    # ========== TREND INDICATORS ==========
    # Moving Averages
    df['ma_20'] = ta.sma(df['Close'], length=20)
    df['ma_50'] = ta.sma(df['Close'], length=50)
    df['ma_200'] = ta.sma(df['Close'], length=200)  # Long-term trend
    df['ema_12'] = ta.ema(df['Close'], length=12)
    df['ema_26'] = ta.ema(df['Close'], length=26)

    # MA Slopes (rate of change)
    df['ma_20_slope'] = df['ma_20'].diff()
    df['ma_50_slope'] = df['ma_50'].diff()

    # MA Cross signals (1 if ma_20 > ma_50, else 0)
    df['ma_cross'] = (df['ma_20'] > df['ma_50']).astype(int)

    # ========== MARKET REGIME DETECTION ==========
    # ADX - Average Directional Index (trend strength)
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx is not None:
        df['adx'] = adx['ADX_14']
        df['plus_di'] = adx['DMP_14']  # +DI
        df['minus_di'] = adx['DMN_14']  # -DI
        # Trend direction signal: +1 bullish, -1 bearish, 0 neutral
        df['di_signal'] = np.where(df['plus_di'] > df['minus_di'], 1,
                                   np.where(df['minus_di'] > df['plus_di'], -1, 0))

    # Trend strength classification
    # Strong trend: ADX > 25, Weak trend: ADX < 20
    df['trend_strength'] = np.where(df['adx'] > 40, 2,  # Very strong
                                    np.where(df['adx'] > 25, 1,  # Strong
                                             np.where(df['adx'] > 20, 0, -1)))  # Weak/Ranging

    # Clear trend direction signal based on multiple factors
    # Combines MA cross, price vs MA200, and DI
    df['trend_direction'] = 0.0
    df.loc[(df['Close'] > df['ma_200']) & (df['ma_20'] > df['ma_50']) & (df['plus_di'] > df['minus_di']), 'trend_direction'] = 1.0  # Bullish
    df.loc[(df['Close'] < df['ma_200']) & (df['ma_20'] < df['ma_50']) & (df['minus_di'] > df['plus_di']), 'trend_direction'] = -1.0  # Bearish

    # Market regime: 1=uptrend, -1=downtrend, 0=ranging
    df['market_regime'] = np.where(
        (df['adx'] > 25) & (df['plus_di'] > df['minus_di']), 1,
        np.where((df['adx'] > 25) & (df['minus_di'] > df['plus_di']), -1, 0)
    )

    # Price momentum (multi-period)
    df['momentum_5'] = df['Close'].pct_change(5)
    df['momentum_20'] = df['Close'].pct_change(20)

    # ========== MOMENTUM INDICATORS ==========
    # RSI
    df['rsi_14'] = ta.rsi(df['Close'], length=14)

    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']

    # Stochastic Oscillator
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    if stoch is not None:
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']

    # Rate of Change (momentum)
    df['roc_10'] = ta.roc(df['Close'], length=10)

    # ========== VOLATILITY INDICATORS ==========
    # ATR
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None:
        # Handle different pandas_ta versions (column names may vary)
        bb_cols = bbands.columns.tolist()
        bb_upper_col = [c for c in bb_cols if c.startswith('BBU')][0]
        bb_middle_col = [c for c in bb_cols if c.startswith('BBM')][0]
        bb_lower_col = [c for c in bb_cols if c.startswith('BBL')][0]

        df['bb_upper'] = bbands[bb_upper_col]
        df['bb_middle'] = bbands[bb_middle_col]
        df['bb_lower'] = bbands[bb_lower_col]
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ========== PRICE ACTION FEATURES ==========
    # Candle body and wick ratios
    df['body_size'] = abs(df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['candle_range'] = df['High'] - df['Low']

    # Price position in daily range
    df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)

    # Returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # ========== DROP NaN VALUES ==========
    df.dropna(inplace=True)

    # Replace any remaining inf values with 0
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # ========== NORMALIZE FEATURES (optional but recommended for RL) ==========
    if normalize:
        # Store original close for reference
        first_close = df['Close'].iloc[0]

        # Normalize price-based columns relative to first close
        price_cols = ['Open', 'High', 'Low', 'Close', 'ma_20', 'ma_50', 'ma_200', 'ema_12', 'ema_26',
                      'bb_upper', 'bb_middle', 'bb_lower']
        for col in price_cols:
            if col in df.columns:
                df[col] = (df[col] / first_close - 1) * 100  # Percentage change from first close

        # Normalize ATR and ADX using z-score
        if 'atr' in df.columns:
            atr_mean = df['atr'].mean()
            atr_std = df['atr'].std()
            df['atr'] = (df['atr'] - atr_mean) / (atr_std + 1e-10)

        # Normalize ADX to [0, 1] range (typical max ~60)
        if 'adx' in df.columns:
            df['adx'] = df['adx'] / 50.0  # Scale to ~[0, 1.2]
        if 'plus_di' in df.columns:
            df['plus_di'] = df['plus_di'] / 50.0
        if 'minus_di' in df.columns:
            df['minus_di'] = df['minus_di'] / 50.0

        if 'ma_20_slope' in df.columns:
            df['ma_20_slope'] = df['ma_20_slope'] / (df['ma_20_slope'].std() + 1e-10)
        if 'ma_50_slope' in df.columns:
            df['ma_50_slope'] = df['ma_50_slope'] / (df['ma_50_slope'].std() + 1e-10)

        # Normalize momentum features
        if 'momentum_5' in df.columns:
            df['momentum_5'] = df['momentum_5'] * 100  # Convert to percentage
        if 'momentum_20' in df.columns:
            df['momentum_20'] = df['momentum_20'] * 100

        # Normalize other features to reasonable ranges
        if 'rsi_14' in df.columns:
            df['rsi_14'] = (df['rsi_14'] - 50) / 50  # Scale to [-1, 1]
        if 'stoch_k' in df.columns:
            df['stoch_k'] = (df['stoch_k'] - 50) / 50
        if 'stoch_d' in df.columns:
            df['stoch_d'] = (df['stoch_d'] - 50) / 50
        if 'bb_pct' in df.columns:
            df['bb_pct'] = (df['bb_pct'] - 0.5) * 2  # Scale to [-1, 1]

        # Clip extreme values
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].clip(-100, 100)

    # Final check for NaN/inf
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df


def get_feature_names() -> list:
    """Returns the list of feature names used in the observation space."""
    return [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'ma_20', 'ma_50', 'ma_200', 'ema_12', 'ema_26', 'ma_20_slope', 'ma_50_slope', 'ma_cross',
        'adx', 'plus_di', 'minus_di', 'di_signal', 'trend_strength', 'trend_direction', 'market_regime',
        'momentum_5', 'momentum_20',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d', 'roc_10',
        'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
        'body_size', 'upper_wick', 'lower_wick', 'candle_range', 'price_position',
        'returns', 'log_returns'
    ]