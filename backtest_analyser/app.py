# app.py
import streamlit as st
import pandas as pd
import os
import re
import plotly.express as px
import ast

# --- Page Configuration ---
st.set_page_config(
    page_title="Diamond Analysis Dashboard",
    page_icon="💎",
    layout="wide"
)

# --- Data Loading Logic (Cached) ---
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'diamond_data', 'backtesting_results'))
BLACKLIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'platinum_data', 'blacklists'))

@st.cache_data(ttl=600)
def get_available_instruments():
    """Scans the results directory to find all available instrument reports."""
    if not os.path.exists(RESULTS_DIR):
        return []
    pattern = re.compile(r"summary_report_(.+)\.csv")
    files = os.listdir(RESULTS_DIR)
    instruments = [pattern.match(f).group(1) for f in files if pattern.match(f)]
    return sorted(instruments)

@st.cache_data(ttl=600)
def load_instrument_data(instrument):
    """Loads the summary and detailed files for a selected instrument."""
    if not instrument:
        return None, None, None

    summary_path = os.path.join(RESULTS_DIR, f"summary_report_{instrument}.csv")
    detailed_path = os.path.join(RESULTS_DIR, f"detailed_report_{instrument}.csv")
    blacklist_path = os.path.join(BLACKLIST_DIR, f"{instrument}.csv")
    
    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    detailed_df = pd.read_csv(detailed_path) if os.path.exists(detailed_path) else pd.DataFrame()
    blacklist_df = pd.read_csv(blacklist_path) if os.path.exists(blacklist_path) else pd.DataFrame()

    return summary_df, detailed_df, blacklist_df

# --- Main App UI ---
st.title("💎 Diamond Analysis Dashboard")
st.markdown("The final validation stage. Use this dashboard to filter for robust strategies and analyze their multi-market performance.")

# --- Sidebar for Controls ---
st.sidebar.title("Controls")
available_instruments = get_available_instruments()

if not available_instruments:
    st.error(f"No result files found. Ensure reports are in: `{RESULTS_DIR}`")
    st.stop()

selected_instrument = st.sidebar.selectbox(
    "Select Instrument Report",
    options=available_instruments,
    index=0
)

# --- Load Data ---
summary_df, detailed_df, blacklist_df = load_instrument_data(selected_instrument)

if summary_df.empty:
    st.warning(f"Could not load summary report for **{selected_instrument}**. The file may be empty or missing.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Strategy Filters")

pf_min = st.sidebar.slider(
    "Minimum Average Profit Factor",
    min_value=0.0, max_value=5.0, value=1.2, step=0.1
)
dd_max = st.sidebar.slider(
    "Maximum Average Drawdown (%)",
    min_value=0.0, max_value=100.0, value=25.0, step=1.0
)
min_markets_passed = st.sidebar.slider(
    "Minimum Markets Passed",
    min_value=0, max_value=summary_df['markets_tested'].max(), value=int(summary_df['markets_tested'].max() / 2), step=1
)
# --- Replace the trades filter section with this updated code ---
# First check if we have either 'total_trades' or individual market trades
trades_column = 'total_trades' if 'total_trades' in summary_df.columns else None

if trades_column:
    min_trades = st.sidebar.slider(
        "Minimum Total Trades",
        min_value=0, 
        max_value=int(summary_df[trades_column].max()), 
        value=50, 
        step=10
    )
    trades_filter = summary_df[trades_column] >= min_trades
else:
    trades_filter = True
    st.sidebar.warning("Trades data not available in summary report")

# --- Update the filters section ---
filtered_summary = summary_df[
    (summary_df['avg_profit_factor'] >= pf_min) &
    (summary_df['avg_max_drawdown_pct'] <= dd_max) &
    (summary_df['markets_passed_count'] >= min_markets_passed) &
    trades_filter
].copy()


# --- Main Content Area with Tabs ---
tab1, tab2, tab3 = st.tabs(["🏆 Strategy Dashboard", "🔬 Detailed Analysis", "🚫 Blacklist"])

# --- TAB 1: Strategy Dashboard ---
with tab1:
    st.header(f"Filtered Strategies for `{selected_instrument}`")
    st.metric("Strategies Found", f"{len(filtered_summary)} / {len(summary_df)}")
    
    st.dataframe(filtered_summary.style.format({
        'avg_profit_factor': '{:.2f}',
        'avg_max_drawdown_pct': '{:.2f}%'
    }))

# --- TAB 2: Detailed Analysis ---
with tab2:
    st.header("Drill-Down Strategy Analysis")
    
    if filtered_summary.empty:
        st.warning("No strategies match the current filter criteria. Adjust the filters in the sidebar to select a strategy.")
    else:
        strategy_id = st.selectbox(
            "Select a Strategy ID to Inspect",
            filtered_summary['strategy_id'].unique()
        )
        
        if strategy_id and not detailed_df.empty:
            strategy_details_all_markets = detailed_df[detailed_df['strategy_id'] == strategy_id]
            
            st.subheader(f"Multi-Market Performance for `{strategy_id}`")
            
            # --- Visualizations: CORRECTED LOGIC ---
            col1, col2 = st.columns(2)
            
            fig_pf = px.bar(
                strategy_details_all_markets,
                x='market',
                y='profit_factor',
                title=f"<b>Profit Factor by Market</b>",
                color='profit_factor',
                color_continuous_scale=px.colors.sequential.Greens
            )
            fig_pf.add_hline(y=1.0, line_dash="dash", line_color="red")
            col1.plotly_chart(fig_pf, use_container_width=True)
            
            fig_dd = px.bar(
                strategy_details_all_markets,
                x='market',
                y='max_drawdown_pct',
                title=f"<b>Max Drawdown (%) by Market</b>",
                color='max_drawdown_pct',
                color_continuous_scale=px.colors.sequential.Reds_r # Reversed
            )
            col2.plotly_chart(fig_dd, use_container_width=True)
            
            # --- Data Table and Regime Analysis ---
            with st.expander("Show Detailed Metrics Table"):
                st.dataframe(strategy_details_all_markets.style.format({
                    'profit_factor': '{:.2f}',
                    'max_drawdown_pct': '{:.2f}%',
                    'final_capital': '${:,.2f}',
                    'win_rate_pct': '{:.2f}%'
                }))

            st.subheader("Market Regime Analysis (Across All Trades)")
            st.info("This shows the percentage of trade entries that occurred under different market conditions, aggregated across all tested markets.")
            
            def parse_dict_col(data_string):
                try: return ast.literal_eval(str(data_string))
                except (ValueError, SyntaxError): return {}

            # Aggregate regime data across all markets for the selected strategy
            agg_regimes = {}
            for col in ['session_pct', 'trend_regime_pct', 'vol_regime_pct']:
                total_counts = {}
                total_trades = 0
                for _, row in strategy_details_all_markets.iterrows():
                    regime_dict = parse_dict_col(row[col])
                    num_trades = row['total_trades']
                    for regime, pct in regime_dict.items():
                        total_counts[regime] = total_counts.get(regime, 0) + (pct / 100 * num_trades)
                    total_trades += num_trades
                
                # Convert back to percentage
                agg_regimes[col] = {k: v / total_trades * 100 for k, v in total_counts.items()} if total_trades > 0 else {}
            
            r_col1, r_col2, r_col3 = st.columns(3)
            if agg_regimes['session_pct']:
                r_col1.plotly_chart(px.pie(names=agg_regimes['session_pct'].keys(), values=agg_regimes['session_pct'].values(), title="<b>Session Entries</b>", hole=.3), use_container_width=True)
            if agg_regimes['trend_regime_pct']:
                r_col2.plotly_chart(px.pie(names=agg_regimes['trend_regime_pct'].keys(), values=agg_regimes['trend_regime_pct'].values(), title="<b>Trend Regime Entries</b>", hole=.3), use_container_width=True)
            if agg_regimes['vol_regime_pct']:
                r_col3.plotly_chart(px.pie(names=agg_regimes['vol_regime_pct'].keys(), values=agg_regimes['vol_regime_pct'].values(), title="<b>Volatility Regime Entries</b>", hole=.3), use_container_width=True)

# --- TAB 3: Blacklist ---
with tab3:
    st.header(f"Blacklisted Strategies for `{selected_instrument}`")
    if blacklist_df.empty:
        st.success("No strategies were blacklisted for this instrument during the last run.")
    else:
        st.write(f"Found {len(blacklist_df)} strategies that failed performance checks.")
        st.dataframe(blacklist_df)