"""
Detailed Performance Monitoring System
=====================================

Pro Tip #2: Keep detailed logs and monitor what works and what doesn't.
This system provides comprehensive analysis of training and testing results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Tuple
import os

class PerformanceMonitor:
    """Comprehensive performance monitoring and analysis."""
    
    def __init__(self, log_dir: str = "performance_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def analyze_training_results(self, model_path: str, results: Dict) -> Dict:
        """Analyze training results and provide insights."""
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'raw_results': results,
            'performance_grade': self._grade_performance(results),
            'insights': self._generate_insights(results),
            'recommendations': self._generate_recommendations(results)
        }
        
        # Save analysis
        analysis_file = f"{self.log_dir}/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
    
    def _grade_performance(self, results: Dict) -> str:
        """Grade the performance from A+ to F."""
        
        win_rate = results.get('win_rate', 0) * 100
        return_pct = results.get('return_pct', 0)
        max_drawdown = results.get('max_drawdown', 1) * 100
        total_trades = results.get('total_trades', 0)
        
        score = 0
        
        # Win rate scoring (40% of grade)
        if win_rate >= 75: score += 40
        elif win_rate >= 65: score += 35
        elif win_rate >= 55: score += 30
        elif win_rate >= 45: score += 20
        elif win_rate >= 35: score += 10
        else: score += 0
        
        # Return scoring (30% of grade)
        if return_pct >= 20: score += 30
        elif return_pct >= 10: score += 25
        elif return_pct >= 5: score += 20
        elif return_pct >= 0: score += 15
        elif return_pct >= -5: score += 10
        else: score += 0
        
        # Drawdown scoring (20% of grade)
        if max_drawdown <= 5: score += 20
        elif max_drawdown <= 10: score += 15
        elif max_drawdown <= 15: score += 10
        elif max_drawdown <= 25: score += 5
        else: score += 0
        
        # Trade frequency scoring (10% of grade)
        if 20 <= total_trades <= 100: score += 10
        elif 10 <= total_trades <= 200: score += 8
        elif 5 <= total_trades <= 300: score += 5
        else: score += 0
        
        # Convert to letter grade
        if score >= 90: return "A+"
        elif score >= 85: return "A"
        elif score >= 80: return "A-"
        elif score >= 75: return "B+"
        elif score >= 70: return "B"
        elif score >= 65: return "B-"
        elif score >= 60: return "C+"
        elif score >= 55: return "C"
        elif score >= 50: return "C-"
        elif score >= 45: return "D"
        else: return "F"
    
    def _generate_insights(self, results: Dict) -> List[str]:
        """Generate insights from the results."""
        
        insights = []
        
        win_rate = results.get('win_rate', 0) * 100
        return_pct = results.get('return_pct', 0)
        max_drawdown = results.get('max_drawdown', 1) * 100
        total_trades = results.get('total_trades', 0)
        profit_factor = results.get('profit_factor', 0)
        
        # Win rate insights
        if win_rate < 30:
            insights.append("🔴 CRITICAL: Win rate extremely low - model may be overfitting or criteria too strict")
        elif win_rate < 45:
            insights.append("🟡 WARNING: Win rate below acceptable threshold - needs improvement")
        elif win_rate > 70:
            insights.append("🟢 EXCELLENT: High win rate achieved")
        
        # Return insights
        if return_pct < -10:
            insights.append("🔴 CRITICAL: Significant losses - model not viable for live trading")
        elif return_pct < 0:
            insights.append("🟡 WARNING: Negative returns - model needs refinement")
        elif return_pct > 15:
            insights.append("🟢 EXCELLENT: Strong positive returns")
        
        # Drawdown insights
        if max_drawdown > 20:
            insights.append("🔴 CRITICAL: Excessive drawdown - risk management insufficient")
        elif max_drawdown > 15:
            insights.append("🟡 WARNING: High drawdown - consider tighter risk controls")
        elif max_drawdown < 8:
            insights.append("🟢 EXCELLENT: Low drawdown - good risk management")
        
        # Trade frequency insights
        if total_trades < 10:
            insights.append("🟡 WARNING: Very few trades - criteria may be too strict")
        elif total_trades > 200:
            insights.append("🟡 WARNING: High trade frequency - may be overtrading")
        
        # Profit factor insights
        if profit_factor < 1.0:
            insights.append("🔴 CRITICAL: Profit factor < 1.0 - losing more than winning")
        elif profit_factor > 2.0:
            insights.append("🟢 EXCELLENT: Strong profit factor")
        
        return insights
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate specific recommendations for improvement."""
        
        recommendations = []
        
        win_rate = results.get('win_rate', 0) * 100
        return_pct = results.get('return_pct', 0)
        max_drawdown = results.get('max_drawdown', 1) * 100
        total_trades = results.get('total_trades', 0)
        
        # Win rate recommendations
        if win_rate < 45:
            recommendations.extend([
                "📈 Increase training timesteps to 2M+",
                "🎯 Relax entry criteria - current thresholds too strict",
                "📊 Add more recent market data (2023-2025)",
                "🔄 Try ensemble learning with multiple models",
                "⚙️ Reduce confidence threshold from 0.85 to 0.70"
            ])
        
        # Return recommendations
        if return_pct < 0:
            recommendations.extend([
                "💰 Increase take-profit targets relative to stop-loss",
                "🛡️ Implement trailing stops to protect profits",
                "📉 Reduce position sizes during drawdown periods",
                "🎲 Add position sizing based on Kelly Criterion"
            ])
        
        # Drawdown recommendations
        if max_drawdown > 15:
            recommendations.extend([
                "🚫 Implement maximum daily loss limits",
                "📊 Add volatility-based position sizing",
                "⏰ Avoid trading during high-impact news events",
                "🔄 Use correlation filters to avoid overexposure"
            ])
        
        # Trade frequency recommendations
        if total_trades < 20:
            recommendations.extend([
                "🎯 Lower confidence requirements for entries",
                "📈 Add more market regimes to trade",
                "⏰ Extend trading hours/sessions",
                "🔄 Reduce minimum time between trades"
            ])
        elif total_trades > 150:
            recommendations.extend([
                "⏰ Increase minimum time between trades",
                "🎯 Raise confidence requirements",
                "📊 Add overtrading penalties to reward function"
            ])
        
        return recommendations
    
    def create_performance_report(self, results: Dict, model_name: str) -> str:
        """Create a comprehensive performance report."""
        
        analysis = self.analyze_training_results(model_name, results)
        
        report = f"""
{'='*80}
PERFORMANCE ANALYSIS REPORT
{'='*80}
Model: {model_name}
Timestamp: {analysis['timestamp']}
Grade: {analysis['performance_grade']}

📊 KEY METRICS:
{'='*50}
Win Rate: {results.get('win_rate', 0)*100:.1f}%
Total Return: {results.get('return_pct', 0):.1f}%
Max Drawdown: {results.get('max_drawdown', 0)*100:.1f}%
Total Trades: {results.get('total_trades', 0)}
Profit Factor: {results.get('profit_factor', 0):.2f}
Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}

💡 INSIGHTS:
{'='*50}
"""
        
        for insight in analysis['insights']:
            report += f"{insight}\n"
        
        report += f"""
🎯 RECOMMENDATIONS:
{'='*50}
"""
        
        for rec in analysis['recommendations']:
            report += f"{rec}\n"
        
        report += f"""
{'='*80}
"""
        
        # Save report
        report_file = f"{self.log_dir}/report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def compare_models(self, results_list: List[Tuple[str, Dict]]) -> str:
        """Compare multiple model results."""
        
        comparison = f"""
{'='*80}
MODEL COMPARISON REPORT
{'='*80}
Timestamp: {datetime.now().isoformat()}

"""
        
        # Create comparison table
        df_data = []
        for model_name, results in results_list:
            df_data.append({
                'Model': model_name,
                'Grade': self._grade_performance(results),
                'Win Rate (%)': f"{results.get('win_rate', 0)*100:.1f}",
                'Return (%)': f"{results.get('return_pct', 0):.1f}",
                'Max DD (%)': f"{results.get('max_drawdown', 0)*100:.1f}",
                'Trades': results.get('total_trades', 0),
                'Profit Factor': f"{results.get('profit_factor', 0):.2f}",
                'Sharpe': f"{results.get('sharpe_ratio', 0):.2f}"
            })
        
        df = pd.DataFrame(df_data)
        comparison += df.to_string(index=False)
        comparison += "\n\n"
        
        # Find best model
        best_model = max(results_list, key=lambda x: self._calculate_score(x[1]))
        comparison += f"🏆 BEST MODEL: {best_model[0]}\n"
        comparison += f"   Grade: {self._grade_performance(best_model[1])}\n"
        comparison += f"   Win Rate: {best_model[1].get('win_rate', 0)*100:.1f}%\n"
        comparison += f"   Return: {best_model[1].get('return_pct', 0):.1f}%\n\n"
        
        comparison += "{'='*80}\n"
        
        # Save comparison
        comp_file = f"{self.log_dir}/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(comp_file, 'w') as f:
            f.write(comparison)
        
        return comparison
    
    def _calculate_score(self, results: Dict) -> float:
        """Calculate numerical score for ranking."""
        win_rate = results.get('win_rate', 0) * 100
        return_pct = results.get('return_pct', 0)
        max_drawdown = results.get('max_drawdown', 1) * 100
        
        # Weighted score
        score = (win_rate * 0.4) + (return_pct * 0.3) + ((20 - max_drawdown) * 0.3)
        return max(0, score)


def analyze_ultra_aggressive_results():
    """Analyze the ultra-aggressive system results."""
    
    # Results from the ultra-aggressive system
    results = {
        'total_trades': 65,
        'win_rate': 0.246,  # 24.6%
        'return_pct': -17.1,
        'max_drawdown': 0.191,  # 19.1%
        'profit_factor': 0.69,
        'sharpe_ratio': -2.66
    }
    
    monitor = PerformanceMonitor()
    report = monitor.create_performance_report(results, "ultra_aggressive_winrate_system")
    
    print(report)
    
    return results, report


if __name__ == "__main__":
    # Analyze the ultra-aggressive results
    results, report = analyze_ultra_aggressive_results()