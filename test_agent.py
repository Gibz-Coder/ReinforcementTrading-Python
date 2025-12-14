import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252*24):
    """Calculate annualized Sharpe ratio for hourly data."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess_returns = np.array(returns) - risk_free_rate / periods_per_year
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve."""
    peak = equity_curve[0]
    max_dd = 0.0
    drawdowns = []

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        drawdowns.append(dd)
        max_dd = max(max_dd, dd)

    return max_dd, drawdowns


def calculate_profit_factor(trades):
    """Calculate profit factor (gross profit / gross loss)."""
    gross_profit = sum(t['pnl_pips'] for t in trades if t['pnl_pips'] > 0)
    gross_loss = abs(sum(t['pnl_pips'] for t in trades if t['pnl_pips'] < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def plot_comprehensive_results(equity_curve, trade_history, test_df, stats, output_dir="./results"):
    """Generate comprehensive visualization of trading results."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig = plt.figure(figsize=(16, 12))

    # 1. Equity Curve
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(equity_curve, label='Equity', color='blue', linewidth=1.5)
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.7, label='Initial Balance')
    ax1.fill_between(range(len(equity_curve)), 10000, equity_curve,
                     where=[e >= 10000 for e in equity_curve], alpha=0.3, color='green')
    ax1.fill_between(range(len(equity_curve)), 10000, equity_curve,
                     where=[e < 10000 for e in equity_curve], alpha=0.3, color='red')
    ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Equity ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown Chart
    ax2 = fig.add_subplot(2, 2, 2)
    max_dd, drawdowns = calculate_max_drawdown(equity_curve)
    ax2.fill_between(range(len(drawdowns)), 0, [-d*100 for d in drawdowns],
                     color='red', alpha=0.5)
    ax2.set_title(f'Drawdown (Max: {max_dd*100:.1f}%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # 3. Trade Distribution
    ax3 = fig.add_subplot(2, 2, 3)
    if trade_history:
        pnls = [t['pnl_pips'] for t in trade_history]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Trade PnL Distribution (Pips)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('PnL (Pips)')
        ax3.grid(True, alpha=0.3)

    # 4. Statistics Summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    stats_text = f"""
    ═══════════════════════════════════════
              TRADING STATISTICS
    ═══════════════════════════════════════

    Total Trades:        {stats.get('total_trades', 0):>10}
    Winning Trades:      {stats.get('winning_trades', 0):>10}
    Losing Trades:       {stats.get('losing_trades', 0):>10}
    Win Rate:            {stats.get('win_rate', 0)*100:>9.1f}%

    ───────────────────────────────────────

    Total PnL (Pips):    {stats.get('total_pnl_pips', 0):>10.1f}
    Total PnL ($):       {stats.get('total_pnl_dollars', 0):>10.2f}
    Avg Win (Pips):      {stats.get('avg_win_pips', 0):>10.1f}
    Avg Loss (Pips):     {stats.get('avg_loss_pips', 0):>10.1f}

    ───────────────────────────────────────

    Final Equity:        ${stats.get('final_equity', 10000):>9,.2f}
    Return:              {stats.get('return_pct', 0):>9.1f}%
    Max Drawdown:        {stats.get('max_drawdown', 0)*100:>9.1f}%
    Profit Factor:       {stats.get('profit_factor', 0):>10.2f}
    Sharpe Ratio:        {stats.get('sharpe_ratio', 0):>10.2f}

    ═══════════════════════════════════════
    """

    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = f"{output_dir}/test_results_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nResults saved to: {output_path}")
    return output_path


def test_agent(
    model_path: str = "models/ppo_forex_eurusd_final",
    test_data_path: str = "data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv",
    window_size: int = 30,
    sl_options: list = None,
    tp_options: list = None,
    output_dir: str = "./results"
):
    """
    Test a trained PPO agent on new data with comprehensive analysis.

    Args:
        model_path: Path to saved model
        test_data_path: Path to test data CSV
        window_size: Observation window size (must match training)
        sl_options: Stop loss options (must match training)
        tp_options: Take profit options (must match training)
        output_dir: Directory for output files
    """
    sl_options = sl_options or [30, 50, 70]
    tp_options = tp_options or [30, 50, 70]

    print("=" * 60)
    print("FOREX TRADING RL AGENT - TESTING")
    print("=" * 60)

    # Load test data
    print(f"\nLoading test data from: {test_data_path}")
    test_df = load_and_preprocess_data(test_data_path, normalize=True)
    print(f"Test data loaded: {len(test_df)} bars, {test_df.shape[1]} features")

    # Create test environment (raw, not wrapped in VecEnv)
    test_env = ForexTradingEnv(
        df=test_df,
        window_size=window_size,
        sl_options=sl_options,
        tp_options=tp_options,
        initial_balance=10000.0,
        risk_per_trade=0.02,
        max_trades_per_day=5,
        render_mode="human"
    )

    # Load trained model
    print(f"Loading model from: {model_path}")
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully!")
    except FileNotFoundError:
        # Try alternative paths
        alt_paths = ["model_eurusd", "models/ppo_forex_eurusd_final.zip",
                     "checkpoints/ppo_forex_eurusd_best_model"]
        for alt_path in alt_paths:
            try:
                model = PPO.load(alt_path)
                print(f"Model loaded from alternative path: {alt_path}")
                break
            except:
                continue
        else:
            raise FileNotFoundError(f"Could not find model at {model_path} or alternatives")

    # Run evaluation using raw environment (not VecEnv)
    print("\nRunning evaluation...")
    obs, info = test_env.reset()
    equity_curve = []
    rewards_history = []
    step_count = 0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        equity_curve.append(test_env.equity)
        rewards_history.append(reward)
        step_count += 1

        # Print progress every 1000 steps
        if step_count % 1000 == 0:
            current_equity = equity_curve[-1]
            print(f"  Step {step_count:5d} | Equity: ${current_equity:,.2f}")

        if terminated or truncated:
            break

    # Get comprehensive statistics
    stats = test_env.get_trade_statistics()
    trade_history = test_env.trade_history

    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nTotal Steps: {step_count}")
    print(f"Total Trades: {stats.get('total_trades', 0)}")
    print(f"Win Rate: {stats.get('win_rate', 0)*100:.1f}%")
    print(f"Total PnL: ${stats.get('total_pnl_dollars', 0):,.2f}")
    print(f"Final Equity: ${stats.get('final_equity', 10000):,.2f}")
    print(f"Return: {stats.get('return_pct', 0):.1f}%")
    print(f"Max Drawdown: {stats.get('max_drawdown', 0)*100:.1f}%")
    print(f"Profit Factor: {stats.get('profit_factor', 0):.2f}")
    print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")

    # Save trade history
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if trade_history:
        trades_df = pd.DataFrame(trade_history)
        trades_output = f"{output_dir}/trade_history_{timestamp}.csv"
        trades_df.to_csv(trades_output, index=False)
        print(f"\nTrade history saved to: {trades_output}")

    # Generate comprehensive visualization
    plot_comprehensive_results(equity_curve, trade_history, test_df, stats, output_dir)

    return stats, trade_history, equity_curve


def main():
    """Main entry point for testing."""
    stats, trades, equity = test_agent(
        model_path="models/ppo_forex_1M",
        test_data_path="data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv",
        window_size=30,
        sl_options=[30, 50, 70],
        tp_options=[30, 50, 70],
        output_dir="./results"
    )


if __name__ == "__main__":
    main()
