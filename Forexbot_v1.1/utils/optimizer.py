"""
/******************************************************************************
 *
 * FILE NAME:           optimizer.py (Save Results)
 *
 * PURPOSE:
 *
 * This version now saves the final ranked results to a CSV file so they
 * can be viewed and analyzed in the dashboard.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 24, 2025
 *
 * VERSION:             33.2 (Save Results)
 *
 ******************************************************************************/
"""
import itertools
import pandas as pd
from types import SimpleNamespace
import configs.config as config
from backtester import Backtester

def run_optimization():
    # ... (the main optimization loop is unchanged) ...
    param_grid = config.OPTIMIZATION_PARAMS
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"--- Starting Optimization ---")
    print(f"Testing {len(param_combinations)} unique parameter combinations...")

    all_results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\nRunning combination {i+1}/{len(param_combinations)}: {params}")
        
        temp_config = SimpleNamespace()
        for setting in dir(config):
            if not setting.startswith('__'):
                setattr(temp_config, setting, getattr(config, setting))
        for key, value in params.items():
            setattr(temp_config, key, value)
            
        backtester = Backtester(temp_config)
        result = backtester.run(
            symbol='EURUSD',
            start_date_str='2023-01-01',
            end_date_str='2024-01-01',
            verbose=False
        )
        
        if result and result.get('total_trades', 0) > 10:
            result['params'] = str(params) # Convert params dict to string for CSV
            all_results.append(result)

    if not all_results:
        print("\n--- Optimization Complete ---")
        print("No profitable combinations found or too few trades executed.")
        return

    results_df = pd.DataFrame(all_results)
    
    results_df['profit_factor_num'] = pd.to_numeric(results_df['profit_factor'], errors='coerce')
    results_df['net_profit_num'] = results_df['net_profit'] # Already a number
    
    ranked_results = results_df.sort_values(by=['profit_factor_num', 'net_profit_num'], ascending=False)

    print("\n\n--- Optimization Complete: Top 5 Results ---")
    # Define columns to display and save
    display_cols = ['params', 'net_profit', 'profit_factor', 'win_rate_percent', 'total_trades', 'max_drawdown_percent']
    print(ranked_results[display_cols].head(5).to_string())

    # --- NEW: Save the results to a file ---
    ranked_results[display_cols].to_csv("optimizer_results.csv", index=False)
    print("\nSaved optimization results to 'optimizer_results.csv'")


if __name__ == '__main__':
    run_optimization()