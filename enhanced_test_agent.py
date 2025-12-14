"""
Enhanced Testing and Analysis Script
===================================

Features:
1. Comprehensive backtesting with enhanced metrics
2. Monte Carlo simulation for risk assessment
3. Market regime analysis
4. Performance attribution analysis
5. Interactive visualizations
6. Risk-adjusted performance metrics
7. Comparison with benchmarks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from enhanced_indicators import load_and_preprocess_data_enhanced
from enhanced_trading_env import EnhancedForexTradingEnv


class EnhancedBacktester:
    """
    Enhanced backtesting engine with comprehensive analysis.
    """
    
    def __init__(self, model_path: str, test_data_path: str, window_size: int = 30):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.window_size = window_size
        
        # Load model and data
        self.model = self._load_model()
        self.test_df = self._load_test_data()
        
        # Results storage
        self.results = {}
        self.trade_history = []
        self.equity_curve = []
        self.drawdown_curve = []
        
    def _load_model(self):
        """Load the trained model."""
        try:
            model = PPO.load(self.model_path)
            print(f"✓ Model loaded from: {self.model_path}")
            return model
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            # Try alternative paths
            alt_paths = [
                "models/enhanced_ppo_forex_final",
                "checkpoints/enhanced_ppo_forex_v2_*/split_0_final",
                "models/ppo_forex_eurusd_final"
            ]
            
            for alt_path in alt_paths:
                try:
                    model = PPO.load(alt_path)
                    print(f"✓ Model loaded from alternative path: {alt_path}")
                    return model
                except:
                    continue
            
            raise FileNotFoundError(f"Could not load model from {self.model_path} or alternatives")
    
    def _load_test_data(self):
        """Load and preprocess test data."""
        print(f"Loading test data from: {self.test_data_path}")
        df = load_and_preprocess_data_enhanced(self.test_data_path, normalize=True)
        print(f"✓ Test data loaded: {len(df)} bars, {df.shape[1]} features")
        return df
    
    def run_backtest(self, render_progress: bool = True) -> Dict:
        """
        Run comprehensive backtest.
        
        Args:
            render_progress: Whether to show progress during backtest
            
        Returns:
            Dictionary with comprehensive results
        """
        print("\n" + "="*60)
        print("RUNNING ENHANCED BACKTEST")
        print("="*60)
        
        # Create test environment
        env = EnhancedForexTradingEnv(
            df=self.test_df,
            window_size=self.window_size,
            initial_balance=10000.0,
            max_trades_per_day=5,
            spread_pips=1.5,
            slippage_pips=0.5,
            render_mode="human" if render_progress else None
        )
        
        # Run backtest
        obs, info = env.reset()
        step_count = 0
        
        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_count += 1
            
            # Show progress
            if render_progress and step_count % 1000 == 0:
                env.render()
            
            if terminated or truncated:
                break
        
        # Get comprehensive results
        self.results = env.get_enhanced_statistics()
        self.trade_history = env.trade_history
        self.equity_curve = env.equity_curve
        
        # Calculate additional metrics
        self._calculate_additional_metrics()
        
        print(f"\n✓ Backtest completed: {step_count} steps, {len(self.trade_history)} trades")
        return self.results
    
    def _calculate_additional_metrics(self):
        """Calculate additional performance metrics."""
        
        if not self.equity_curve:
            return
        
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Risk metrics
        self.results['volatility'] = np.std(returns) * np.sqrt(252 * 24)  # Annualized
        self.results['var_95'] = np.percentile(returns, 5)
        self.results['var_99'] = np.percentile(returns, 1)
        self.results['cvar_95'] = np.mean(returns[returns <= self.results['var_95']])
        
        # Performance ratios
        self.results['calmar_ratio'] = (self.results.get('return_pct', 0) / 100) / max(self.results.get('max_drawdown', 0.01), 0.01)
        
        # Trade analysis
        if self.trade_history:
            self._analyze_trades()
        
        # Drawdown analysis
        self._calculate_drawdown_metrics()
    
    def _analyze_trades(self):
        """Analyze individual trades."""
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # Duration analysis
        self.results['avg_trade_duration'] = trades_df['bars_held'].mean()
        self.results['max_trade_duration'] = trades_df['bars_held'].max()
        
        # PnL distribution
        pnls = trades_df['pnl_pips'].values
        self.results['pnl_skewness'] = pd.Series(pnls).skew()
        self.results['pnl_kurtosis'] = pd.Series(pnls).kurtosis()
        
        # Consecutive wins/losses
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for pnl in pnls:
            if pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
        
        self.results['max_consecutive_wins'] = max_win_streak
        self.results['max_consecutive_losses'] = max_loss_streak
        
        # Direction analysis
        long_trades = trades_df[trades_df['direction'] == 'LONG']
        short_trades = trades_df[trades_df['direction'] == 'SHORT']
        
        if len(long_trades) > 0:
            self.results['long_win_rate'] = (long_trades['pnl_pips'] > 0).mean()
            self.results['long_avg_pnl'] = long_trades['pnl_pips'].mean()
        
        if len(short_trades) > 0:
            self.results['short_win_rate'] = (short_trades['pnl_pips'] > 0).mean()
            self.results['short_avg_pnl'] = short_trades['pnl_pips'].mean()
    
    def _calculate_drawdown_metrics(self):
        """Calculate detailed drawdown metrics."""
        
        if not self.equity_curve:
            return
        
        equity_array = np.array(self.equity_curve)
        peak = equity_array[0]
        drawdowns = []
        
        for equity in equity_array:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            drawdowns.append(dd)
        
        self.drawdown_curve = drawdowns
        
        # Drawdown statistics
        drawdown_array = np.array(drawdowns)
        self.results['avg_drawdown'] = np.mean(drawdown_array[drawdown_array > 0])
        self.results['drawdown_duration'] = self._calculate_drawdown_duration(drawdown_array)
        self.results['recovery_factor'] = self.results.get('return_pct', 0) / max(self.results.get('max_drawdown', 0.01) * 100, 0.01)
    
    def _calculate_drawdown_duration(self, drawdowns: np.ndarray) -> float:
        """Calculate average drawdown duration."""
        
        in_drawdown = False
        durations = []
        current_duration = 0
        
        for dd in drawdowns:
            if dd > 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_drawdown:
                    durations.append(current_duration)
                    in_drawdown = False
                    current_duration = 0
        
        return np.mean(durations) if durations else 0
    
    def monte_carlo_analysis(self, n_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation for risk assessment.
        
        Args:
            n_simulations: Number of simulations to run
            
        Returns:
            Dictionary with Monte Carlo results
        """
        print(f"\nRunning Monte Carlo analysis ({n_simulations} simulations)...")
        
        if not self.trade_history:
            print("No trade history available for Monte Carlo analysis")
            return {}
        
        # Extract trade returns
        trade_returns = [t['pnl_dollars'] for t in self.trade_history]
        
        # Run simulations
        final_equities = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Randomly resample trades with replacement
            simulated_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Calculate equity curve
            equity_curve = [10000]  # Starting balance
            peak = 10000
            max_dd = 0
            
            for ret in simulated_returns:
                new_equity = equity_curve[-1] + ret
                equity_curve.append(new_equity)
                
                if new_equity > peak:
                    peak = new_equity
                
                dd = (peak - new_equity) / peak
                max_dd = max(max_dd, dd)
            
            final_equities.append(equity_curve[-1])
            max_drawdowns.append(max_dd)
        
        # Calculate statistics
        mc_results = {
            'final_equity_mean': np.mean(final_equities),
            'final_equity_std': np.std(final_equities),
            'final_equity_5th': np.percentile(final_equities, 5),
            'final_equity_95th': np.percentile(final_equities, 95),
            'prob_positive': np.mean(np.array(final_equities) > 10000),
            'prob_loss_10pct': np.mean(np.array(final_equities) < 9000),
            'prob_loss_20pct': np.mean(np.array(final_equities) < 8000),
            'max_dd_mean': np.mean(max_drawdowns),
            'max_dd_95th': np.percentile(max_drawdowns, 95),
            'max_dd_99th': np.percentile(max_drawdowns, 99)
        }
        
        print("✓ Monte Carlo analysis completed")
        return mc_results
    
    def regime_analysis(self) -> Dict:
        """
        Analyze performance by market regime.
        
        Returns:
            Dictionary with regime-based performance
        """
        print("\nAnalyzing performance by market regime...")
        
        if not self.trade_history:
            return {}
        
        # Group trades by market regime
        regime_performance = {}
        
        for trade in self.trade_history:
            regime = trade.get('market_regime', 0)
            
            if regime not in regime_performance:
                regime_performance[regime] = {
                    'trades': [],
                    'total_pnl': 0,
                    'wins': 0,
                    'losses': 0
                }
            
            regime_performance[regime]['trades'].append(trade)
            regime_performance[regime]['total_pnl'] += trade['pnl_pips']
            
            if trade['pnl_pips'] > 0:
                regime_performance[regime]['wins'] += 1
            else:
                regime_performance[regime]['losses'] += 1
        
        # Calculate statistics for each regime
        regime_stats = {}
        regime_names = {
            -2: 'Strong Downtrend',
            -1: 'Weak Downtrend', 
             0: 'Ranging Market',
             1: 'Weak Uptrend',
             2: 'Strong Uptrend'
        }
        
        for regime, data in regime_performance.items():
            total_trades = len(data['trades'])
            
            if total_trades > 0:
                regime_stats[regime_names.get(regime, f'Regime {regime}')] = {
                    'total_trades': total_trades,
                    'win_rate': data['wins'] / total_trades,
                    'avg_pnl': data['total_pnl'] / total_trades,
                    'total_pnl': data['total_pnl'],
                    'profit_factor': abs(sum(t['pnl_pips'] for t in data['trades'] if t['pnl_pips'] > 0) / 
                                       min(-1, sum(t['pnl_pips'] for t in data['trades'] if t['pnl_pips'] < 0)))
                }
        
        print("✓ Regime analysis completed")
        return regime_stats
    
    def create_comprehensive_report(self, output_dir: str = "./results") -> str:
        """
        Create comprehensive HTML report with all analysis.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Run all analyses
        mc_results = self.monte_carlo_analysis()
        regime_results = self.regime_analysis()
        
        # Create visualizations
        self._create_visualizations(output_dir, timestamp)
        
        # Generate HTML report
        report_path = f"{output_dir}/enhanced_backtest_report_{timestamp}.html"
        
        html_content = self._generate_html_report(mc_results, regime_results, timestamp)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"\n✓ Comprehensive report saved to: {report_path}")
        return report_path
    
    def _create_visualizations(self, output_dir: str, timestamp: str):
        """Create comprehensive visualizations."""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Main Performance Dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # Equity curve
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(self.equity_curve, color='blue', linewidth=2)
        plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.7)
        plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = plt.subplot(3, 3, 2)
        plt.fill_between(range(len(self.drawdown_curve)), 0, 
                        [-dd*100 for dd in self.drawdown_curve], 
                        color='red', alpha=0.5)
        plt.title('Drawdown (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # Trade PnL distribution
        if self.trade_history:
            ax3 = plt.subplot(3, 3, 3)
            pnls = [t['pnl_pips'] for t in self.trade_history]
            plt.hist(pnls, bins=30, alpha=0.7, color='green', edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title('Trade PnL Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('PnL (Pips)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Cumulative PnL
            ax4 = plt.subplot(3, 3, 4)
            cumulative_pnl = np.cumsum(pnls)
            plt.plot(cumulative_pnl, color='purple', linewidth=2)
            plt.title('Cumulative PnL (Pips)', fontsize=14, fontweight='bold')
            plt.xlabel('Trade Number')
            plt.ylabel('Cumulative PnL (Pips)')
            plt.grid(True, alpha=0.3)
            
            # Trade duration distribution
            ax5 = plt.subplot(3, 3, 5)
            durations = [t['bars_held'] for t in self.trade_history]
            plt.hist(durations, bins=20, alpha=0.7, color='orange', edgecolor='black')
            plt.title('Trade Duration Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Duration (Hours)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Win/Loss by direction
            ax6 = plt.subplot(3, 3, 6)
            long_trades = [t for t in self.trade_history if t['direction'] == 'LONG']
            short_trades = [t for t in self.trade_history if t['direction'] == 'SHORT']
            
            directions = ['Long', 'Short']
            win_rates = [
                sum(1 for t in long_trades if t['pnl_pips'] > 0) / max(len(long_trades), 1),
                sum(1 for t in short_trades if t['pnl_pips'] > 0) / max(len(short_trades), 1)
            ]
            
            plt.bar(directions, [wr*100 for wr in win_rates], color=['blue', 'red'], alpha=0.7)
            plt.title('Win Rate by Direction', fontsize=14, fontweight='bold')
            plt.ylabel('Win Rate (%)')
            plt.grid(True, alpha=0.3)
        
        # Monthly returns heatmap (if enough data)
        if len(self.equity_curve) > 720:  # More than 30 days
            ax7 = plt.subplot(3, 3, 7)
            monthly_returns = self._calculate_monthly_returns()
            if monthly_returns:
                sns.heatmap(monthly_returns, annot=True, fmt='.1f', cmap='RdYlGn', center=0)
                plt.title('Monthly Returns (%)', fontsize=14, fontweight='bold')
        
        # Risk metrics
        ax8 = plt.subplot(3, 3, 8)
        risk_metrics = [
            ('Sharpe Ratio', self.results.get('sharpe_ratio', 0)),
            ('Calmar Ratio', self.results.get('calmar_ratio', 0)),
            ('Recovery Factor', self.results.get('recovery_factor', 0)),
            ('Profit Factor', self.results.get('profit_factor', 0))
        ]
        
        metrics, values = zip(*risk_metrics)
        colors = ['green' if v > 1 else 'red' if v < 0 else 'orange' for v in values]
        plt.barh(metrics, values, color=colors, alpha=0.7)
        plt.title('Risk-Adjusted Metrics', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Performance summary text
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_text = f"""
        PERFORMANCE SUMMARY
        ═══════════════════════════════════
        
        Total Return:        {self.results.get('return_pct', 0):>8.1f}%
        Total Trades:        {self.results.get('total_trades', 0):>8d}
        Win Rate:            {self.results.get('win_rate', 0)*100:>8.1f}%
        
        Max Drawdown:        {self.results.get('max_drawdown', 0)*100:>8.1f}%
        Avg Drawdown:        {self.results.get('avg_drawdown', 0)*100:>8.1f}%
        
        Sharpe Ratio:        {self.results.get('sharpe_ratio', 0):>8.2f}
        Profit Factor:       {self.results.get('profit_factor', 0):>8.2f}
        
        Volatility:          {self.results.get('volatility', 0)*100:>8.1f}%
        VaR (95%):           {self.results.get('var_95', 0)*100:>8.1f}%
        """
        
        ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_dashboard_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualizations created")
    
    def _calculate_monthly_returns(self) -> Optional[pd.DataFrame]:
        """Calculate monthly returns for heatmap."""
        
        if len(self.equity_curve) < 720:  # Less than 30 days
            return None
        
        # Convert to daily returns (assuming hourly data)
        daily_equity = [self.equity_curve[i] for i in range(0, len(self.equity_curve), 24)]
        daily_returns = np.diff(daily_equity) / np.array(daily_equity[:-1]) * 100
        
        # Create date index (approximate)
        dates = pd.date_range(start='2023-01-01', periods=len(daily_returns), freq='D')
        returns_series = pd.Series(daily_returns, index=dates)
        
        # Group by month and year
        monthly_returns = returns_series.groupby([returns_series.index.year, returns_series.index.month]).sum()
        
        # Reshape for heatmap
        monthly_df = monthly_returns.unstack(level=1)
        monthly_df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        return monthly_df
    
    def _generate_html_report(self, mc_results: Dict, regime_results: Dict, timestamp: str) -> str:
        """Generate comprehensive HTML report."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Forex Trading Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .neutral {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Enhanced Forex Trading Backtest Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <strong>Total Return:</strong> 
                    <span class="{'positive' if self.results.get('return_pct', 0) > 0 else 'negative'}">
                        {self.results.get('return_pct', 0):.1f}%
                    </span>
                </div>
                <div class="metric">
                    <strong>Sharpe Ratio:</strong> 
                    <span class="{'positive' if self.results.get('sharpe_ratio', 0) > 1 else 'neutral' if self.results.get('sharpe_ratio', 0) > 0 else 'negative'}">
                        {self.results.get('sharpe_ratio', 0):.2f}
                    </span>
                </div>
                <div class="metric">
                    <strong>Max Drawdown:</strong> 
                    <span class="negative">{self.results.get('max_drawdown', 0)*100:.1f}%</span>
                </div>
                <div class="metric">
                    <strong>Win Rate:</strong> 
                    <span class="{'positive' if self.results.get('win_rate', 0) > 0.5 else 'negative'}">
                        {self.results.get('win_rate', 0)*100:.1f}%
                    </span>
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Trades</td><td>{self.results.get('total_trades', 0)}</td></tr>
                    <tr><td>Winning Trades</td><td>{self.results.get('winning_trades', 0)}</td></tr>
                    <tr><td>Losing Trades</td><td>{self.results.get('losing_trades', 0)}</td></tr>
                    <tr><td>Win Rate</td><td>{self.results.get('win_rate', 0)*100:.1f}%</td></tr>
                    <tr><td>Profit Factor</td><td>{self.results.get('profit_factor', 0):.2f}</td></tr>
                    <tr><td>Average Win</td><td>{self.results.get('avg_win_pips', 0):.1f} pips</td></tr>
                    <tr><td>Average Loss</td><td>{self.results.get('avg_loss_pips', 0):.1f} pips</td></tr>
                    <tr><td>Max Consecutive Wins</td><td>{self.results.get('max_consecutive_wins', 0)}</td></tr>
                    <tr><td>Max Consecutive Losses</td><td>{self.results.get('max_consecutive_losses', 0)}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Risk Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Maximum Drawdown</td><td>{self.results.get('max_drawdown', 0)*100:.1f}%</td></tr>
                    <tr><td>Average Drawdown</td><td>{self.results.get('avg_drawdown', 0)*100:.1f}%</td></tr>
                    <tr><td>Volatility (Annualized)</td><td>{self.results.get('volatility', 0)*100:.1f}%</td></tr>
                    <tr><td>VaR (95%)</td><td>{self.results.get('var_95', 0)*100:.1f}%</td></tr>
                    <tr><td>CVaR (95%)</td><td>{self.results.get('cvar_95', 0)*100:.1f}%</td></tr>
                    <tr><td>Calmar Ratio</td><td>{self.results.get('calmar_ratio', 0):.2f}</td></tr>
                    <tr><td>Recovery Factor</td><td>{self.results.get('recovery_factor', 0):.2f}</td></tr>
                </table>
            </div>
        """
        
        # Add Monte Carlo results if available
        if mc_results:
            html += f"""
            <div class="section">
                <h2>Monte Carlo Analysis</h2>
                <p>Based on {1000} simulations of trade resampling:</p>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Probability of Profit</td><td>{mc_results.get('prob_positive', 0)*100:.1f}%</td></tr>
                    <tr><td>Probability of 10% Loss</td><td>{mc_results.get('prob_loss_10pct', 0)*100:.1f}%</td></tr>
                    <tr><td>Probability of 20% Loss</td><td>{mc_results.get('prob_loss_20pct', 0)*100:.1f}%</td></tr>
                    <tr><td>Expected Final Equity</td><td>${mc_results.get('final_equity_mean', 0):,.0f}</td></tr>
                    <tr><td>5th Percentile Outcome</td><td>${mc_results.get('final_equity_5th', 0):,.0f}</td></tr>
                    <tr><td>95th Percentile Outcome</td><td>${mc_results.get('final_equity_95th', 0):,.0f}</td></tr>
                    <tr><td>Expected Max Drawdown</td><td>{mc_results.get('max_dd_mean', 0)*100:.1f}%</td></tr>
                    <tr><td>95th Percentile Max DD</td><td>{mc_results.get('max_dd_95th', 0)*100:.1f}%</td></tr>
                </table>
            </div>
            """
        
        # Add regime analysis if available
        if regime_results:
            html += """
            <div class="section">
                <h2>Performance by Market Regime</h2>
                <table>
                    <tr><th>Market Regime</th><th>Trades</th><th>Win Rate</th><th>Avg PnL</th><th>Total PnL</th><th>Profit Factor</th></tr>
            """
            
            for regime, stats in regime_results.items():
                html += f"""
                    <tr>
                        <td>{regime}</td>
                        <td>{stats['total_trades']}</td>
                        <td>{stats['win_rate']*100:.1f}%</td>
                        <td>{stats['avg_pnl']:.1f} pips</td>
                        <td>{stats['total_pnl']:.1f} pips</td>
                        <td>{stats['profit_factor']:.2f}</td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        html += """
            <div class="section">
                <h2>Visualizations</h2>
                <p>Performance dashboard and detailed charts have been saved as PNG files in the results directory.</p>
            </div>
            
            <div class="section">
                <h2>Disclaimer</h2>
                <p><strong>Important:</strong> This backtest is based on historical data and does not guarantee future performance. 
                Past performance is not indicative of future results. Trading involves substantial risk and may not be suitable for all investors.</p>
            </div>
        </body>
        </html>
        """
        
        return html


def main():
    """Main function to run enhanced backtesting."""
    
    # Configuration
    model_path = "models/enhanced_ppo_forex_final"  # Update with your model path
    test_data_path = "data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv"
    
    try:
        # Create backtester
        backtester = EnhancedBacktester(model_path, test_data_path, window_size=30)
        
        # Run backtest
        results = backtester.run_backtest(render_progress=True)
        
        # Print summary
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"Total Return:     {results.get('return_pct', 0):>8.1f}%")
        print(f"Total Trades:     {results.get('total_trades', 0):>8d}")
        print(f"Win Rate:         {results.get('win_rate', 0)*100:>8.1f}%")
        print(f"Profit Factor:    {results.get('profit_factor', 0):>8.2f}")
        print(f"Sharpe Ratio:     {results.get('sharpe_ratio', 0):>8.2f}")
        print(f"Max Drawdown:     {results.get('max_drawdown', 0)*100:>8.1f}%")
        print(f"Calmar Ratio:     {results.get('calmar_ratio', 0):>8.2f}")
        
        # Generate comprehensive report
        report_path = backtester.create_comprehensive_report()
        print(f"\nComprehensive report generated: {report_path}")
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()