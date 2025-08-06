# run_validation.py - Simple script to run validation
"""
Run comprehensive validation of your RL models
"""

from rl_backtesting_framework import run_complete_validation

if __name__ == "__main__":
    # Run validation on your trained models
    results = run_complete_validation()
    
    # The system will:
    # 1. Load your trained models from ./models/
    # 2. Run realistic backtesting with transaction costs
    # 3. Perform walk-forward analysis 
    # 4. Generate comprehensive reports
    # 5. Create interactive visualizations
