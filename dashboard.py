"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           dashboard.py
 *
 * PURPOSE:
 *
 * This module creates a web-based, interactive dashboard for monitoring
 * the trading bot and visualizing backtest results using the Streamlit
 * library.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             4.0
 *
 ******************************************************************************/
"""
import streamlit as st
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Forex Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Main App ---
st.title("📈 Algorithmic Forex Trading Bot Dashboard")

st.sidebar.header("Controls")
app_mode = st.sidebar.selectbox(
    "Choose Dashboard Mode",
    ["Backtest Results", "Live Monitor (Placeholder)"]
)

# --- Backtesting Mode ---
if app_mode == "Backtest Results":
    st.header("Backtest Performance Analysis")

    # Check if result files exist
    if os.path.exists("backtest_results.csv") and os.path.exists("equity_curve.csv"):
        
        # Load the data
        results_df = pd.read_csv("backtest_results.csv")
        equity_curve_df = pd.read_csv("equity_curve.csv")

        if results_df.empty:
            st.warning("The backtest ran successfully but no trades were executed.")
        else:
            # --- Performance Metrics ---
            st.subheader("Key Performance Indicators (KPIs)")
            
            total_trades = len(results_df)
            wins = results_df[results_df['profit'] > 0]
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            total_profit = results_df['profit'].sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Net Profit", f"${total_profit:,.2f}")
            col2.metric("Total Trades", f"{total_trades}")
            col3.metric("Win Rate", f"{win_rate:.2f}%")

            # --- Equity Curve Chart ---
            st.subheader("Equity Curve")
            st.line_chart(equity_curve_df)

            # --- Trade Log ---
            st.subheader("Trade Log")
            st.dataframe(results_df)

    else:
        st.info("Please run the `backtester.py` script first to generate result files.")

# --- Live Monitor Placeholder ---
elif app_mode == "Live Monitor (Placeholder)":
    st.header("Live Trading Monitor")
    st.info("This section is a placeholder for a future live monitoring feature.")
    
    st.subheader("Current Status")
    st.success("Bot is running. Analyzing markets...")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Account Equity", "$100,000.00", "0%")
    col2.metric("Open Positions", "0", "")
    col3.metric("Today's P/L", "$0.00", "")