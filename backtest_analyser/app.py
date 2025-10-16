# app.py (V10 - Final Version with Market Trend Context Chart)

import streamlit as st
import pandas as pd
import os
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import numpy as np

# --- Page Configuration & Constants ---
st.set_page_config(page_title="Strategy Post-Mortem", page_icon="🕵️", layout="wide")
SHARPE_CAP = 10.0

# --- Path Configuration ---
CORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIAMOND_RESULTS_DIR = os.path.join(CORE_DIR, 'diamond_data', 'backtesting_results')
ZIRCON_RESULTS_DIR = os.path.join(CORE_DIR, 'zircon_data', 'results')
BLACKLIST_DIR = os.path.join(CORE_DIR, 'platinum_data', 'blacklists')
PREPARED_DATA_DIR = os.path.join(CORE_DIR, 'diamond_data', 'prepared_data')
TRADE_LOGS_DIR = os.path.join(CORE_DIR, 'zircon_data', 'trade_logs') # CORRECTED PATH

# --- Data Loading & Helper Functions ---
@st.cache_data(ttl=600)
def get_available_reports():
    if not os.path.exists(ZIRCON_RESULTS_DIR): return []
    pattern = re.compile(r"summary_report_(.+)\.csv")
    files = os.listdir(ZIRCON_RESULTS_DIR)
    reports = [pattern.match(f).group(1) for f in files if pattern.match(f)]
    return sorted(reports)

@st.cache_data(ttl=600)
def load_and_merge_data(report_name):
    # This function is unchanged and correct
    if not report_name: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    zircon_summary_path = os.path.join(ZIRCON_RESULTS_DIR, f"summary_report_{report_name}.csv")
    zircon_detailed_path = os.path.join(ZIRCON_RESULTS_DIR, f"detailed_report_{report_name}.csv")
    if not os.path.exists(zircon_summary_path): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    zircon_summary_df = pd.read_csv(zircon_summary_path)
    zircon_detailed_df = pd.read_csv(zircon_detailed_path)
    diamond_report_path = os.path.join(DIAMOND_RESULTS_DIR, f"diamond_report_{report_name}.csv")
    mastery_df = pd.read_csv(diamond_report_path) if os.path.exists(diamond_report_path) else pd.DataFrame()
    zircon_renamed = zircon_summary_df.rename(columns={'avg_profit_factor': 'validation_avg_pf', 'avg_sharpe_ratio': 'validation_avg_sharpe', 'avg_max_drawdown_pct': 'validation_avg_dd_pct', 'total_trades': 'validation_total_trades', 'validation_markets_passed': 'validation_passed'})
    if not mastery_df.empty:
        mastery_renamed = mastery_df.rename(columns={'profit_factor': 'mastery_pf', 'sharpe_ratio': 'mastery_sharpe', 'max_drawdown_pct': 'mastery_dd_pct', 'total_trades': 'mastery_total_trades'})
        master_view_df = pd.merge(zircon_renamed, mastery_renamed[['strategy_id', 'mastery_pf', 'mastery_sharpe', 'mastery_dd_pct', 'mastery_total_trades']], on='strategy_id', how='left')
    else:
        master_view_df = zircon_renamed.copy()
        for col in ['mastery_pf', 'mastery_sharpe', 'mastery_dd_pct', 'mastery_total_trades']: master_view_df[col] = np.nan
    for col in ['mastery_pf', 'validation_avg_pf']:
        if col in master_view_df.columns: master_view_df[col] = master_view_df[col].replace(np.inf, 999)
    for col in ['mastery_sharpe', 'validation_avg_sharpe']:
        if col in master_view_df.columns: master_view_df[col] = master_view_df[col].replace(np.inf, SHARPE_CAP).clip(upper=SHARPE_CAP)
    return master_view_df, zircon_detailed_df, mastery_df

def write_to_blacklist(strategy_blueprint, report_name):
    blacklist_path = os.path.join(BLACKLIST_DIR, f"{report_name}.csv")
    key_cols = ['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin']
    blueprint_to_add = pd.DataFrame([strategy_blueprint[key_cols].to_dict()])
    try: blacklist_df = pd.read_csv(blacklist_path)
    except FileNotFoundError: blacklist_df = pd.DataFrame(columns=key_cols)
    updated_blacklist = pd.concat([blacklist_df, blueprint_to_add], ignore_index=True).drop_duplicates()
    updated_blacklist.to_csv(blacklist_path, index=False)
    st.toast(f"Strategy blueprint blacklisted in {report_name}.csv!"); st.cache_data.clear()

@st.cache_data(ttl=3600)
def load_market_internals(markets):
    internals = []
    for market_name in markets:
        # FIX: Load from the correct prepared_data directory and use parquet
        silver_path = os.path.join(PREPARED_DATA_DIR, f"{market_name.replace('.csv','')}_silver.parquet")
        if os.path.exists(silver_path):
            df = pd.read_parquet(silver_path, columns=['trend_regime', 'vol_regime', 'BB_width', 'close'])
            trend_pct = df['trend_regime'].value_counts(normalize=True).get('trend', 0) * 100
            vol_pct = df['vol_regime'].value_counts(normalize=True).get('high_vol', 0) * 100
            avg_bbw = df['BB_width'].mean()
            price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            internals.append({'market': market_name, '% Time in Trend': trend_pct, '% Time High Vol': vol_pct, 'Avg BB Width': avg_bbw, 'Overall Price Change %': price_change})
    return pd.DataFrame(internals)

def parse_dict_col(data_string):
    try: return ast.literal_eval(str(data_string))
    except: return {}

@st.cache_data(ttl=3600)
def load_full_silver_data(market_name):
    silver_path = os.path.join(PREPARED_DATA_DIR, f"{market_name.replace('.csv','')}_silver.parquet")
    if os.path.exists(silver_path):
        df = pd.read_parquet(silver_path, columns=['time', 'close'])
        df['time'] = pd.to_datetime(df['time'])
        return df
    return None

@st.cache_data(ttl=600)
def load_trade_log(strategy_id, market_name):
    """Loads the detailed trade log for a specific strategy and market."""
    log_path = os.path.join(TRADE_LOGS_DIR, strategy_id, f"{market_name.replace('.csv','')}.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        return df
    return None

def parse_dict_col(data_string):
    try: return ast.literal_eval(str(data_string))
    except: return {}

# --- UI START ---
st.title("🕵️ Strategy Post-Mortem Dashboard")
st.sidebar.title("Controls")
available_reports = get_available_reports()
if not available_reports: st.error(f"No Zircon result files found in `{ZIRCON_RESULTS_DIR}`"); st.stop()

selected_report = st.sidebar.selectbox("Select Report to Analyze", options=available_reports, key="report_selector")
master_df, detailed_df, mastery_df = load_and_merge_data(selected_report)
if master_df.empty: st.warning(f"Could not load data for **{selected_report}**."); st.stop()

# Re-instated Sidebar Filters
st.sidebar.header("Filter Strategies")
if st.sidebar.button("Reset All Filters"):
    st.session_state.m_pf, st.session_state.m_sh, st.session_state.v_pf = 0.0, 0.0, 0.0
    st.session_state.v_pass = '0/0'; st.rerun()
st.sidebar.markdown("**Mastery Filters (Origin Market)**")
mastery_pf_min = st.sidebar.slider("Min Mastery PF", 0.0, 10.0, 1.5, 0.1, key='m_pf')
mastery_sharpe_min = st.sidebar.slider("Min Mastery Sharpe", 0.0, SHARPE_CAP, 1.0, 0.1, key='m_sh')
st.sidebar.markdown("---")
st.sidebar.markdown("**Validation Filters (Other Markets)**")
validation_pf_min = st.sidebar.slider("Min Validation Avg PF", 0.0, 5.0, 1.0, 0.1, key='v_pf')
pass_options = sorted([opt for opt in master_df['validation_passed'].unique() if pd.notna(opt)], key=lambda x: (int(x.split('/')[1]), int(x.split('/')[0])), reverse=True)
if not pass_options: pass_options = ['0/0']
min_markets_passed_str = st.sidebar.selectbox("Min Validation Markets Passed", options=pass_options, key='v_pass')

# Apply Filters
min_passed_count = int(min_markets_passed_str.split('/')[0])
filtered_df = master_df.copy()
for col in ['mastery_pf', 'mastery_sharpe', 'validation_avg_pf', 'validation_markets_passed_count']: filtered_df[col] = filtered_df[col].fillna(0)
filtered_df = filtered_df[(filtered_df['mastery_pf'] >= mastery_pf_min) & (filtered_df['mastery_sharpe'] >= mastery_sharpe_min) & (filtered_df['validation_avg_pf'] >= validation_pf_min) & (filtered_df['validation_markets_passed_count'] >= min_passed_count)]

# --- Main Content with Tabs ---
tab1, tab2 = st.tabs(["🏆 Strategy Dashboard", "🕵️ Post-Mortem Analysis"])

with tab1:
    # This tab is restored and correct
    st.header(f"Strategy Dashboard for `{selected_report}`")
    st.info("Use the sidebar filters to discover robust strategies. The table below shows the survivors.")
    st.metric("Robust Strategies Found", f"{len(filtered_df)} / {len(master_df)}")
    display_cols = {'strategy_id': 'ID', 'mastery_pf': 'Mastery PF', 'mastery_sharpe': 'Mastery Sharpe', 'validation_avg_pf': 'Validation PF', 'validation_passed': 'Passed', 'market_rule': 'Market Rule'}
    page_size = st.number_input("Strategies per page:", 10, 100, 20, 5, key="page_size")
    total_pages = max(1, (len(filtered_df) - 1) // page_size + 1)
    current_page = st.number_input("Page:", 1, total_pages, 1, 1, key="page_num")
    start_idx, end_idx = (current_page - 1) * page_size, current_page * page_size
    def format_df_display(df):
        df_display = df.copy()
        for col in ['mastery_pf', 'validation_avg_pf']:
            if col in df_display: df_display[col] = df_display[col].apply(lambda x: "💯 Wins" if x >= 999 else f"{x:.2f}")
        return df_display
    paginated_df = filtered_df.iloc[start_idx:end_idx]
    formatted_paginated_df = format_df_display(paginated_df)
    cols_to_show = [col for col in display_cols.keys() if col in formatted_paginated_df.columns]
    st.dataframe(formatted_paginated_df[cols_to_show].rename(columns=display_cols).style.format({'Mastery Sharpe': '{:.2f}'}))

with tab2:
    st.header("Post-Mortem Analysis")
    if filtered_df.empty:
        st.warning("No strategies match the current filters. Relax the filters in the sidebar to select a strategy for analysis.")
    else:
        strategy_id = st.selectbox("Select a Strategy for Interrogation", filtered_df['strategy_id'].unique())
        if strategy_id:
            strategy_data = filtered_df.loc[filtered_df['strategy_id'] == strategy_id].iloc[0]
            validation_data = detailed_df[detailed_df['strategy_id'] == strategy_id].copy()
            mastery_data_row = mastery_df[mastery_df['strategy_id'] == strategy_id]
            mastery_data = mastery_data_row.iloc[0] if not mastery_data_row.empty else None

            # --- Section 1: Suspect Profile (Unchanged) ---
            st.subheader("1. Strategy Profile")
            trade_type = strategy_data.get('trade_type', 'N/A').upper()
            sl_def = f"{strategy_data['sl_def']:.3f}%" if isinstance(strategy_data['sl_def'], float) else f"{strategy_data['sl_def']} (Bin {strategy_data['sl_bin'] or 'N/A'})".replace('.0', '')
            tp_def = f"{strategy_data['tp_def']:.3f}%" if isinstance(strategy_data['tp_def'], float) else f"{strategy_data['tp_def']} (Bin {strategy_data['tp_bin'] or 'N/A'})".replace('.0', '')
            card_col1, card_col2, card_col3 = st.columns(3)
            card_col1.metric("Trade Type", trade_type); card_col2.metric("Stop-Loss Logic", sl_def); card_col3.metric("Take-Profit Logic", tp_def)
            st.code(strategy_data['market_rule'], language='sql')
            st.sidebar.markdown("---"); st.sidebar.header("Manual Actions")
            if st.sidebar.button("🚫 Manually Blacklist this Blueprint", type="primary", key=f"bl_{strategy_id}"):
                write_to_blacklist(strategy_data, selected_report); st.rerun()

            # --- Section 2: The Verdict ---
            st.subheader("2. The Verdict: Was the Edge Real?")
            v_col1, v_col2 = st.columns(2)
            # This section is unchanged and correct
            with v_col1:
                st.markdown(f"#### Mastery Performance (on `{selected_report}`)")
                if mastery_data is not None:
                    pf_display = "💯 Wins" if mastery_data['profit_factor'] >= 999 else f"{mastery_data['profit_factor']:.2f}"
                    sharpe_display = f"{min(mastery_data['sharpe_ratio'], SHARPE_CAP):.2f}" + ("+" if mastery_data['sharpe_ratio'] > SHARPE_CAP else "")
                    st.metric("Profit Factor", pf_display); st.metric("Sharpe Ratio", sharpe_display); st.metric("Max Drawdown", f"{mastery_data['max_drawdown_pct']:.2f}%")
                else: st.warning("Mastery data not found.")
            with v_col2:
                st.markdown("#### Validation Performance (Avg on Others)")
                pf_display_val = "💯 Wins" if strategy_data['validation_avg_pf'] >= 999 else f"{strategy_data['validation_avg_pf']:.2f}"
                sharpe_display_val = f"{strategy_data['validation_avg_sharpe']:.2f}"
                st.metric("Avg Profit Factor", pf_display_val); st.metric("Avg Sharpe Ratio", sharpe_display_val); st.metric("Markets Passed", strategy_data['validation_passed'])

            # --- Section 3: Performance Breakdown ---
            st.subheader("3. Evidence: Performance Breakdown")
            metric_to_plot = st.radio("Select Metric to Compare:", ["Profit Factor", "Sharpe Ratio", "Max Drawdown %"], horizontal=True, key="metric_radio")
            metric_map = {'Profit Factor': 'profit_factor', 'Sharpe Ratio': 'sharpe_ratio', 'Max Drawdown %': 'max_drawdown_pct'}
            selected_metric = metric_map[metric_to_plot]
            if mastery_data is not None: full_perf_data = pd.concat([pd.DataFrame([mastery_data]), validation_data], ignore_index=True)
            else: full_perf_data = validation_data.copy()
            full_perf_data.replace(np.inf, np.nan, inplace=True)
            fig_compare = px.bar(full_perf_data, x='market', y=selected_metric, color=selected_metric, title=f"<b>{metric_to_plot} Across All Tested Markets</b>", color_continuous_scale='RdYlGn' if selected_metric != 'max_drawdown_pct' else 'RdYlGn_r')
            fig_compare.add_hline(y=1.0 if selected_metric == 'profit_factor' else 0.0, line_dash="dash", line_color="white")
            st.plotly_chart(fig_compare, use_container_width=True)

            # --- NEW: Sections 4, 5, 6: Per-Market Deep Dive ---
            
            # Combine all data for this deep dive
            if mastery_data is not None:
                full_perf_data = pd.concat([pd.DataFrame([mastery_data]), validation_data], ignore_index=True)
            else:
                full_perf_data = validation_data.copy()

            regime_cols_exist = 'session_pct' in full_perf_data.columns
            if not regime_cols_exist:
                st.warning("Regime columns (e.g., 'session_pct') not found in detailed reports. Cannot perform deep dive analysis.")
            else:
                # Loop through each regime type to create a dedicated section
                for regime_type in ['session', 'trend_regime', 'vol_regime']:
                    st.subheader(f"4. Deep Dive: {regime_type.replace('_', ' ').title()} Performance")
                    st.info(f"Analyze the strategy's performance within each {regime_type.replace('_regime','')} category, market by market.")
                    
                    # Create a dataframe from the regime dictionaries
                    regime_data = []
                    for _, row in full_perf_data.iterrows():
                        regime_dict = parse_dict_col(row[f'{regime_type}_pct'])
                        for regime, pct in regime_dict.items():
                            regime_data.append({
                                'market': row['market'],
                                'Regime': regime,
                                'Trade Percentage': pct,
                                'Profit Factor': row['profit_factor']
                            })
                    regime_df = pd.DataFrame(regime_data)
                    regime_df['Profit Factor'] = regime_df['Profit Factor'].replace(np.inf, 5) # Cap for color

                    # Create a faceted bar chart
                    fig = px.bar(regime_df, 
                                 x='Regime', 
                                 y='Trade Percentage',
                                 color='Profit Factor',
                                 color_continuous_scale='RdYlGn',
                                 range_color=[0, 2], # Set a consistent color scale centered on 1
                                 facet_col='market',
                                 labels={'Trade Percentage': '% of Trades in this Regime'},
                                 title=f"<b>Performance by {regime_type.replace('_', ' ').title()} in Each Market</b>")
                    fig.update_xaxes(matches=None)
                    st.plotly_chart(fig, use_container_width=True)

            # --- Section 5: Market Internals (Unchanged) ---
            st.subheader("5. Market Internals: Did the Strategy Get Lucky?")
            all_markets = full_perf_data['market'].unique()
            market_internals_df = load_market_internals(all_markets)
            if not market_internals_df.empty:
                internals_vs_perf = pd.merge(market_internals_df, full_perf_data[['market', 'profit_factor']], on='market', how='left').fillna(0)
                internal_col1, internal_col2 = st.columns(2)
                with internal_col1:
                    fig_internal1 = px.scatter(internals_vs_perf, x='% Time in Trend', y='profit_factor', color='market', title="Performance vs. Market's Tendency to Trend", hover_name='market')
                    st.plotly_chart(fig_internal1, use_container_width=True)
                with internal_col2:
                    fig_internal2 = px.scatter(internals_vs_perf, x='Overall Price Change %', y='profit_factor', color='market', title="Performance vs. Market's Overall Direction", hover_name='market')
                    st.plotly_chart(fig_internal2, use_container_width=True)
            else:
                st.warning(f"Could not load Silver data to analyze market internals. Check path: `{PREPARED_DATA_DIR}`")
            
            # --- NEW/UPGRADED Section 6: Market Trend Context ---
            st.subheader("6. Performance in Context: Market Trend")
            st.info("Visualize individual trades (Green=Win, Red=Loss) overlaid on the market's price history to see how the strategy performs in different macro trends.")

            all_markets_for_strat = []
            if mastery_data is not None: all_markets_for_strat.append(mastery_data['market'])
            all_markets_for_strat.extend(validation_data['market'].unique())

            for market_name in sorted(list(set(all_markets_for_strat))):
                st.markdown(f"#### Trend Context for `{market_name}`")
                
                silver_df = load_full_silver_data(market_name)
                trade_log_df = load_trade_log(strategy_id, market_name)

                if silver_df is None:
                    st.warning(f"Could not load Silver data for {market_name}."); continue
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=silver_df['time'], y=silver_df['close'], mode='lines', name='Close Price', line=dict(color='rgba(128, 128, 128, 0.5)')))

                if trade_log_df is not None:
                    wins = trade_log_df[trade_log_df['pnl'] > 0]
                    losses = trade_log_df[trade_log_df['pnl'] <= 0]
                    
                    fig.add_trace(go.Scatter(
                        x=wins['entry_time'], y=wins['entry_price'],
                        mode='markers', name='Winning Trades',
                        marker=dict(color='limegreen', size=8, symbol='triangle-up', line=dict(width=1, color='DarkSlateGrey'))
                    ))
                    fig.add_trace(go.Scatter(
                        x=losses['entry_time'], y=losses['entry_price'],
                        mode='markers', name='Losing Trades',
                        marker=dict(color='red', size=8, symbol='triangle-down', line=dict(width=1, color='DarkSlateGrey'))
                    ))
                else:
                    st.info(f"Trade log not found for this market. Run the Zircon Validator to generate it.")
                
                fig.update_layout(title=f"Trade Entries vs. Market Trend for {market_name}", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
