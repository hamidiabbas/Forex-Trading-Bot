"""
/******************************************************************************
 *
 * FILE NAME:           dashboard.py (Optimizer View)
 *
 * PURPOSE:
 *
 * This version adds a new "Optimization Results" mode to the dashboard,
 * allowing for easy viewing and analysis of the optimizer's output.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 24, 2025
 *
 * VERSION:             39.0 (Optimizer View)
 *
 ******************************************************************************/
"""
import streamlit as st
import pandas as pd
import os
from performance_analyzer import PerformanceAnalyzer

# --- Page Configuration ---
st.set_page_config(page_title="Forex Bot Dashboard", page_icon="🤖", layout="wide")

# --- Main App ---
st.title("🤖 Algorithmic Forex Trading Bot Dashboard")
st.sidebar.header("Controls")
app_mode = st.sidebar.selectbox(
    "Choose Dashboard Mode",
    ["Backtest Analysis", "Optimization Results", "Live Monitor (Placeholder)"]
)

# --- Backtesting Mode ---
if app_mode == "Backtest Analysis":
    st.header("Backtest Performance Analysis")
    # TODO: Implement backtest analysis display here.
    st.info("Backtest analysis display is not implemented in this version.")

# --- NEW: Optimization Results Mode ---
elif app_mode == "Optimization Results":
    st.header("Strategy Optimization Results")
    
    results_file = "optimizer_results.csv"
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        st.info("Showing the best-performing parameter combinations, sorted by Profit Factor and Net Profit.")
        
        # Format columns for better display
        df['net_profit'] = df['net_profit'].apply(lambda x: f"${x:,.2f}")
        df['profit_factor'] = df['profit_factor'].apply(lambda x: f"{x:.2f}")
        df['win_rate_percent'] = df['win_rate_percent'].apply(lambda x: f"{x:.2f}%")
        df['max_drawdown_percent'] = df['max_drawdown_percent'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No optimization results found. Please run `optimizer.py` first.")


# --- Live Monitor Placeholder ---
elif app_mode == "Live Monitor (Placeholder)":
    st.header("Live Trading Monitor")
    st.info("This section is a placeholder for a future live monitoring feature.")