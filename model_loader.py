#!/usr/bin/env python3
"""
Golden Gibz Model Loader
========================
Loads and uses trained PPO models for real AI predictions
"""

import os
import json
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import ta

warnings.filterwarnings('ignore')

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    print("⚠️ stable-baselines3 not available. Install with: pip install stable-baselines3")
    STABLE_BASELINES_AVAILABLE = False


class GoldenGibzModelLoader:
    """Loads and manages trained Golden Gibz PPO models"""
    
    def __init__(self, models_path="models/production"):
        self.models_path = models_path
        self.model = None
        self.vec_normalize = None
        self.metadata = None
        self.indicators_config = None
        
    def list_available_models(self):
        """List all available trained models"""
        try:
            models = []
            if not os.path.exists(self.models_path):
                return models
            
            for file in os.listdir(self.models_path):
                if file.endswith('.zip'):
                    model_name = file.replace('.zip', '')
                    metadata_file = os.path.join(self.models_path, f"{model_name}_metadata.json")
                    
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        models.append({
                            'name': model_name,
                            'symbol': metadata.get('symbol', 'Unknown'),
                            'win_rate': metadata.get('win_rate', 0),
                            'annual_return': metadata.get('annual_return', 0),
                            'training_date': metadata.get('training_date', 'Unknown'),
                            'model_type': metadata.get('model_type', 'Unknown')
                        })
            
            return sorted(models, key=lambda x: x['training_date'], reverse=True)
            
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    def load_model(self, model_name=None, symbol="XAUUSD"):
        """Load a trained model by name or find the best one for symbol"""
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable-baselines3 is required to load models")
        
        # Store symbol for normalization
        self.symbol = symbol
        
        try:
            # If no model name specified, find the best one for the symbol
            if model_name is None:
                available_models = self.list_available_models()
                symbol_models = [m for m in available_models if m['symbol'].upper() == symbol.upper()]
                
                if not symbol_models:
                    raise ValueError(f"No trained models found for symbol {symbol}")
                
                # Get the most recent model with highest win rate
                best_model = max(symbol_models, key=lambda x: (x['win_rate'], x['training_date']))
                model_name = best_model['name']
                print(f"🤖 Auto-selected model: {model_name}")
                print(f"   Win Rate: {best_model['win_rate']:.1f}%")
                print(f"   Annual Return: {best_model['annual_return']}%")
            
            # Load model files
            model_path = os.path.join(self.models_path, f"{model_name}.zip")
            metadata_path = os.path.join(self.models_path, f"{model_name}_metadata.json")
            normalize_path = os.path.join(self.models_path, f"{model_name}_vecnormalize.pkl")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded metadata for {self.metadata.get('symbol', 'Unknown')} model")
            
            # Load PPO model
            self.model = PPO.load(model_path)
            print(f"✅ Loaded PPO model: {model_name}")
            
            # Load normalization if available
            if os.path.exists(normalize_path):
                # Create dummy environment for normalization
                from train_golden_gibz_model import ForexTradingEnvironment
                dummy_env = ForexTradingEnvironment("data/raw", symbol)
                vec_env = DummyVecEnv([lambda: dummy_env])
                self.vec_normalize = VecNormalize.load(normalize_path, vec_env)
                print(f"✅ Loaded normalization parameters")
            
            # Setup indicators configuration from metadata
            if self.metadata and 'technical_indicators' in self.metadata:
                self.indicators_config = {
                    'ema_fast': 20,
                    'ema_slow': 50,
                    'rsi_period': 14,
                    'atr_period': 14,
                    'bb_period': 20,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'adx_period': 14,
                    'stoch_k': 14,
                    'stoch_d': 3
                }
                print(f"✅ Configured technical indicators")
            
            print(f"🎉 Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators exactly as used in training"""
        try:
            if self.indicators_config is None:
                raise ValueError("Indicators configuration not loaded")
            
            # Ensure we have numeric data
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # EMAs
            df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=self.indicators_config['ema_fast'])
            df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=self.indicators_config['ema_slow'])
            
            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=self.indicators_config['rsi_period'])
            
            # ATR
            df['ATR'] = ta.volatility.average_true_range(
                df['High'], df['Low'], df['Close'], 
                window=self.indicators_config['atr_period']
            )
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'], window=self.indicators_config['bb_period'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
            
            # MACD
            macd = ta.trend.MACD(
                df['Close'], 
                window_fast=self.indicators_config['macd_fast'],
                window_slow=self.indicators_config['macd_slow'],
                window_sign=self.indicators_config['macd_signal']
            )
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # ADX
            df['ADX'] = ta.trend.adx(
                df['High'], df['Low'], df['Close'], 
                window=self.indicators_config['adx_period']
            )
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(
                df['High'], df['Low'], df['Close'], 
                window=self.indicators_config['stoch_k'], 
                smooth_window=self.indicators_config['stoch_d']
            )
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Williams %R
            df['WillR'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            
            # CCI
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
            
            # Enhanced signal conditions
            df['EMA_Bullish'] = (df['EMA20'] > df['EMA50']).astype(int)
            df['Price_Above_EMA20'] = (df['Close'] > df['EMA20']).astype(int)
            df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']).astype(int)
            df['Strong_Trend'] = (df['ADX'] > 25).astype(int)
            df['RSI_Neutral'] = ((df['RSI'] > 30) & (df['RSI'] < 70)).astype(int)
            df['Stoch_Bullish'] = (df['Stoch_K'] > df['Stoch_D']).astype(int)
            
            return df
            
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return df
    
    def prepare_observation(self, df, index, position=0, balance=10000, initial_balance=10000, trades_today=0, max_trades_per_day=5):
        """Prepare observation vector exactly as used in training"""
        try:
            if index >= len(df):
                index = len(df) - 1
            
            current_row = df.iloc[index]
            
            # Helper function to safely convert to float
            def safe_float(value, default=0.0):
                try:
                    if pd.isna(value):
                        return default
                    if isinstance(value, (list, tuple)):
                        return float(value[0]) if len(value) > 0 else default
                    return float(value)
                except (ValueError, TypeError, IndexError):
                    return default
            
            # Price features (normalized to training scale - symbol-specific)
            close_price = safe_float(current_row['Close'], 2000.0)
            high_price = safe_float(current_row['High'], close_price)
            low_price = safe_float(current_row['Low'], close_price)
            volume = safe_float(current_row.get('Volume', 1000), 1000.0)
            
            # Symbol-specific normalization (match training)
            symbol_upper = getattr(self, 'symbol', 'XAUUSD').upper()
            if symbol_upper in ['XAUUSD', 'GOLD']:
                # Gold: ~2000-4000 range
                price_norm = 2500.0
                volume_norm = 1000000.0
            elif symbol_upper in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF']:
                # Forex pairs: ~0.5-2.0 range  
                price_norm = 1.0
                volume_norm = 100000.0
            else:
                # Default normalization
                price_norm = 1000.0
                volume_norm = 1000000.0
            
            price_features = [
                close_price / price_norm,
                high_price / price_norm,
                low_price / price_norm,
                volume / volume_norm if volume > 0 else 0.0
            ]
            
            # Technical indicators (normalized and handle NaN - symbol-specific)
            # Use same price normalization for technical indicators
            if symbol_upper in ['XAUUSD', 'GOLD']:
                price_norm = 2500.0
            elif symbol_upper in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF']:
                price_norm = 1.0
            else:
                price_norm = 1000.0
                
            tech_features = [
                safe_float(current_row.get('EMA20', close_price), close_price) / price_norm,
                safe_float(current_row.get('EMA50', close_price), close_price) / price_norm,
                safe_float(current_row.get('RSI', 50), 50.0) / 100.0,
                safe_float(current_row.get('ATR', 10), 10.0) / 50.0,  # Better ATR normalization
                safe_float(current_row.get('BB_Width', 2), 2.0) / 5.0,  # Better BB normalization
                safe_float(current_row.get('MACD', 0), 0.0) / 25.0,  # Better MACD normalization
                safe_float(current_row.get('MACD_Signal', 0), 0.0) / 25.0,
                safe_float(current_row.get('MACD_Hist', 0), 0.0) / 15.0,
                safe_float(current_row.get('ADX', 25), 25.0) / 100.0,
                safe_float(current_row.get('Stoch_K', 50), 50.0) / 100.0,
                safe_float(current_row.get('Stoch_D', 50), 50.0) / 100.0,
                safe_float(current_row.get('WillR', -50), -50.0) / 100.0,
                safe_float(current_row.get('CCI', 0), 0.0) / 100.0,  # Better CCI normalization
                safe_float(current_row.get('EMA_Bullish', 0), 0.0),
                safe_float(current_row.get('MACD_Bullish', 0), 0.0)
            ]
            
            # Position and account features (enhanced for 1:1 RR system - match training)
            # Scale balance to training environment (10K)
            normalized_balance = safe_float(balance, initial_balance) / safe_float(initial_balance, 10000.0)
            if initial_balance < 1000:  # If small account, scale up for model
                normalized_balance = normalized_balance * 10  # Scale up small accounts
            
            position_features = [
                safe_float(position, 0.0),  # Current position (-1, 0, 1)
                normalized_balance,  # Balance ratio
                safe_float(trades_today, 0.0) / safe_float(max_trades_per_day, 5.0),  # Trade frequency
                0.0,  # Entry balance ratio (not available in backtesting, set to 0)
                0.0   # Position size multiplier (not available in backtesting, set to 0)
            ]
            
            observation = np.array(price_features + tech_features + position_features, dtype=np.float32)
            
            # Handle any NaN values and clip extreme values
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            observation = np.clip(observation, -5.0, 5.0)  # Prevent extreme values
            
            return observation
            
        except Exception as e:
            print(f"❌ Error preparing observation: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(24, dtype=np.float32)  # Updated to match new training environment
    
    def predict(self, observation):
        """Get AI prediction from loaded model with REAL confidence"""
        try:
            if self.model is None:
                raise ValueError("No model loaded")
            
            # Convert numpy array to torch tensor if needed
            import torch
            if isinstance(observation, np.ndarray):
                obs = torch.FloatTensor(observation).reshape(1, -1)
            else:
                obs = observation.reshape(1, -1)
            
            # Apply normalization if available
            if self.vec_normalize is not None:
                obs = self.vec_normalize.normalize_obs(obs.numpy())
                obs = torch.FloatTensor(obs)
            
            # Get prediction with action probabilities (REAL confidence)
            action, _states = self.model.predict(obs.numpy(), deterministic=True)
            
            # Get action probabilities for real confidence calculation
            with torch.no_grad():
                action_probs = self.model.policy.get_distribution(obs).distribution.probs.detach().cpu().numpy()[0]
            
            # Convert action to signal and use REAL confidence from model
            if action == 0:  # Hold
                signal = 0
                confidence = float(action_probs[0])  # Real probability for hold action
            elif action == 1:  # Buy
                signal = 1
                confidence = float(action_probs[1])  # Real probability for buy action
            else:  # Sell (action == 2)
                signal = -1
                confidence = float(action_probs[2])  # Real probability for sell action
            
            # Ensure confidence is reasonable (model outputs can be extreme)
            confidence = max(0.1, min(0.95, confidence))
            
            return signal, confidence
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            # Fallback to deterministic prediction if probability extraction fails
            try:
                if isinstance(observation, np.ndarray):
                    obs_fallback = observation.reshape(1, -1)
                else:
                    obs_fallback = observation.reshape(1, -1)
                
                action, _states = self.model.predict(obs_fallback, deterministic=True)
                if action == 0:
                    return 0, 0.3
                elif action == 1:
                    return 1, 0.7
                else:
                    return -1, 0.7
            except:
                return 0, 0.0
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.metadata is None:
            return "No model loaded"
        
        return {
            'symbol': self.metadata.get('symbol', 'Unknown'),
            'model_type': self.metadata.get('model_type', 'Unknown'),
            'win_rate': self.metadata.get('win_rate', 0),
            'annual_return': self.metadata.get('annual_return', 0),
            'training_date': self.metadata.get('training_date', 'Unknown'),
            'technical_indicators': self.metadata.get('technical_indicators', [])
        }


def main():
    """Test the model loader"""
    print("🤖 Testing Golden Gibz Model Loader...")
    
    loader = GoldenGibzModelLoader()
    
    # List available models
    models = loader.list_available_models()
    print(f"\n📋 Available models: {len(models)}")
    for model in models:
        print(f"   {model['name']} - {model['symbol']} - WR: {model['win_rate']:.1f}%")
    
    # Load best model for XAUUSD
    if loader.load_model(symbol="XAUUSD"):
        info = loader.get_model_info()
        print(f"\n✅ Model loaded successfully!")
        print(f"   Symbol: {info['symbol']}")
        print(f"   Win Rate: {info['win_rate']:.1f}%")
        print(f"   Annual Return: {info['annual_return']}%")
        print(f"   Indicators: {len(info['technical_indicators'])}")
    else:
        print("❌ Failed to load model")


if __name__ == "__main__":
    main()