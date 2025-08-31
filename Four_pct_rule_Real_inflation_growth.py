# This file considers real data for SP500 and inflation.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
from datetime import datetime, timedelta
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="4% Rule Simulation with Real Data")

st.title("4% Rule Simulation with Real Historical Data")
st.markdown("""
This tool calculates the evolution of a portfolio starting with your initial balance, 
with annual withdrawals split monthly, adjusted for real historical inflation, 
and portfolio growth according to the S&P500 Total Return Index.  
Data is fetched live from Yahoo Finance and Federal Reserve sources.
""")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_cpi_date_range():
    """Determine the earliest available CPI data date"""
    try:
        # Test fetch to determine available date range for US CPI
        test_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL&cosd=1950-01-01"
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            test_df = pd.read_csv(pd.io.common.StringIO(response.text))
            
            # Handle different date column names
            date_col = None
            if 'observation_date' in test_df.columns:
                date_col = 'observation_date'
            elif 'DATE' in test_df.columns:
                date_col = 'DATE'
            
            if date_col and len(test_df) > 0:
                # Remove rows with missing data (marked as '.')
                value_col = [col for col in test_df.columns if col != date_col][0]
                test_df = test_df[test_df[value_col] != '.']
                if len(test_df) > 0:
                    earliest_date = pd.to_datetime(test_df[date_col].iloc[0])
                    return earliest_date
        
        # Default fallback
        return pd.to_datetime('1970-01-01')
        
    except Exception as e:
        st.warning(f"Could not determine CPI date range: {e}")
        return pd.to_datetime('1970-01-01')

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_sp500_data(start_date):
    """Fetch S&P 500 total return data from Yahoo Finance"""
    try:
        # Use S&P 500 Total Return Index (^SP500TR) for more accurate results
        # Fallback to regular S&P 500 if total return index isn't available
        tickers = ['^GSPC', '^SP500TR']
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = datetime.now().strftime('%Y-%m-%d')
        
        for ticker in tickers:
            try:
                # Set auto_adjust=False to avoid the warning and for consistency
                sp500 = yf.download(ticker, start=start_str, end=end_str, 
                                  progress=False, auto_adjust=False)
                
                if not sp500.empty and len(sp500) > 100:  # Ensure we have substantial data
                    # Use adjusted close for better accuracy
                    if 'Adj Close' in sp500.columns:
                        sp500_data = sp500[['Adj Close']].copy()
                        sp500_data.columns = ['SP500']
                    else:
                        sp500_data = sp500[['Close']].copy()
                        sp500_data.columns = ['SP500']
                    
                    # Ensure timezone-naive dates
                    if sp500_data.index.tz is not None:
                        sp500_data.index = sp500_data.index.tz_localize(None)
                    
                    return sp500_data, ticker
                    
            except Exception as ticker_error:
                st.warning(f"Failed to fetch {ticker}: {ticker_error}")
                continue
        
        raise Exception("Could not fetch S&P 500 data from any ticker")
        
    except Exception as e:
        st.error(f"Error fetching S&P 500 data: {str(e)}")
        return None, None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_cpi_data(start_date):
    """Fetch CPI data from FRED API"""
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        
        # FRED API endpoints with dynamic start date
        us_cpi_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL&cosd={start_str}"
        can_cpi_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=CANCPIALLMINMEI&cosd={start_str}"
        
        # Fetch US CPI
        us_response = requests.get(us_cpi_url, timeout=15)
        if us_response.status_code != 200:
            raise Exception(f"Failed to fetch US CPI data: HTTP {us_response.status_code}")
            
        us_cpi_text = us_response.text
        us_cpi = pd.read_csv(pd.io.common.StringIO(us_cpi_text))
        
        # Handle different possible date column names
        date_col = None
        if 'observation_date' in us_cpi.columns:
            date_col = 'observation_date'
        elif 'DATE' in us_cpi.columns:
            date_col = 'DATE'
        elif 'date' in us_cpi.columns:
            date_col = 'date'
        
        if date_col is None:
            raise Exception(f"No date column found in US CPI data. Available columns: {list(us_cpi.columns)}")
        
        us_cpi[date_col] = pd.to_datetime(us_cpi[date_col])
        us_cpi = us_cpi.set_index(date_col)
        
        # Clean the data - remove '.' values and convert to float
        value_col = [col for col in us_cpi.columns if col not in ['observation_date', 'DATE', 'date']][0]
        us_cpi[value_col] = us_cpi[value_col].replace('.', np.nan)
        us_cpi[value_col] = pd.to_numeric(us_cpi[value_col], errors='coerce')
        us_cpi = us_cpi.dropna()
        us_cpi.columns = ['US_CPI']
        
        # Fetch Canada CPI
        can_response = requests.get(can_cpi_url, timeout=15)
        if can_response.status_code != 200:
            st.warning(f"Failed to fetch Canadian CPI data: HTTP {can_response.status_code}")
            # Continue with just US data
            return us_cpi
        
        can_cpi_text = can_response.text
        can_cpi = pd.read_csv(pd.io.common.StringIO(can_cpi_text))
        
        if 'observation_date' in can_cpi.columns or 'DATE' in can_cpi.columns:
            date_col_can = 'observation_date' if 'observation_date' in can_cpi.columns else 'DATE'
            can_cpi[date_col_can] = pd.to_datetime(can_cpi[date_col_can])
            can_cpi = can_cpi.set_index(date_col_can)
            
            # Clean the data
            can_value_col = [col for col in can_cpi.columns if col not in ['observation_date', 'DATE', 'date']][0]
            can_cpi[can_value_col] = can_cpi[can_value_col].replace('.', np.nan)
            can_cpi[can_value_col] = pd.to_numeric(can_cpi[can_value_col], errors='coerce')
            can_cpi = can_cpi.dropna()
            can_cpi.columns = ['CAN_CPI']
            
            # Combine CPI data
            cpi_data = us_cpi.join(can_cpi, how='outer')
        else:
            cpi_data = us_cpi
            cpi_data['CAN_CPI'] = np.nan  # Add empty Canadian data
        
        cpi_data = cpi_data.sort_index().dropna(how='all')
        
        return cpi_data
        
    except Exception as e:
        st.error(f"Error fetching CPI data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_and_align_data():
    """Load and align all data sources with dynamic date range"""
    
    # Step 1: Determine the earliest CPI data available
    earliest_cpi_date = fetch_cpi_date_range()
    
    # Step 2: Fetch CPI data from the earliest available date
    cpi_data = fetch_cpi_data(earliest_cpi_date)
    
    if cpi_data is None:
        return None, None, None, None
    
    # Step 3: Get the actual start date from CPI data
    actual_start_date = cpi_data.index[0]
    
    # Step 4: Fetch S&P 500 data from the same start date
    sp500_data, sp500_ticker = fetch_sp500_data(actual_start_date)
    
    if sp500_data is None:
        return None, None, None, None
    
    # Step 5: Align data by joining on dates
    merged = sp500_data.join(cpi_data, how='inner')
    merged = merged.sort_index().dropna()
    
    if len(merged) < 24:  # At least 2 years of data
        st.error(f"Not enough overlapping data. Only {len(merged)} months available.")
        return None, None, None, None
    
    # Step 6: Resample to end-of-month and forward fill any gaps
    merged = merged.resample('M').last().fillna(method='ffill').dropna()
    
    return merged, sp500_ticker, len(merged), actual_start_date

# Load data
df, sp500_ticker, total_months, data_start_date = load_and_align_data()

if df is not None:
    # Check if we have Canadian CPI data
    has_canadian_cpi = 'CAN_CPI' in df.columns and not df['CAN_CPI'].isna().all()
    
    if not has_canadian_cpi:
        st.warning("⚠️ Canadian CPI data not available. Only US CPI will be used.")
    
    # Sidebar
    country_options = ["US"]
    if has_canadian_cpi:
        country_options.append("Canada")
    
    country = st.sidebar.selectbox("Country CPI for Inflation Adjustment", country_options)
    initial_balance = st.sidebar.number_input("Initial Portfolio Balance ($)", value=1000000, step=10000, min_value=1000)
    withdraw_rate = st.sidebar.slider("Annual Withdrawal Rate (%)", 1.0, 10.0, 4.0, 0.1)
    
    # Date range selection
    st.sidebar.subheader("Simulation Period")
    min_year = df.index[0].year
    max_year = df.index[-1].year
    
    # Ensure we have at least 25 years available for max range
    max_selectable_end = min(max_year, min_year + 50)  # Cap at 50 years to avoid too long simulations
    
    start_year = st.sidebar.slider("Start Year", min_year, max_year - 1, 
                                  value=max(min_year, max_year - 35), 
                                  help="Select the starting year for your simulation")
    
    end_year = st.sidebar.slider("End Year", start_year + 1, max_selectable_end, 
                                value=min(start_year + 25, max_selectable_end),
                                help="Select the ending year for your simulation")
    
    years = end_year - start_year
    st.sidebar.info(f"Simulation length: {years} years")
    
    # Choose CPI series
    cpi_col = 'US_CPI' if country == "US" else 'CAN_CPI'
    
    # Calculate monthly returns
    inflation_series = df[cpi_col].pct_change().fillna(0)
    growth_series = df['SP500'].pct_change().fillna(0)
    
    # Filter data to selected date range
    start_date = pd.to_datetime(f"{start_year}-01-01")
    end_date = pd.to_datetime(f"{end_year}-12-31")
    
    # Get data within the selected range
    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered_df = df[mask]
    
    if len(filtered_df) < 12:
        st.error(f"Not enough data in selected range {start_year}-{end_year}. Please select a different range.")
        st.stop()
    
    # Use filtered data for simulation
    inflation_monthly = inflation_series[mask]
    growth_monthly = growth_series[mask]
    n_months = len(inflation_monthly)
    
    # 4% Rule Simulation
    balance = initial_balance
    base_initial_monthly_withdraw = initial_balance * (withdraw_rate / 100) / 12  # never changes
    monthly_withdraw = base_initial_monthly_withdraw  # this one updates with inflation
    
    balances = []
    cum_withdrawals = []
    monthly_withdrawals = []
    total_withdrawn = 0
    
    # Create aligned date range
    simulation_dates = inflation_monthly.index
    
    for i in range(n_months):
        # Record current state
        balances.append(balance)
        monthly_withdrawals.append(monthly_withdraw)
        
        # Make withdrawal
        total_withdrawn += monthly_withdraw
        cum_withdrawals.append(total_withdrawn)
        
        # Apply market growth and subtract withdrawal
        balance = balance * (1 + growth_monthly.iloc[i]) - monthly_withdraw
        
        # Adjust withdrawal for inflation annually (every 12 months)
        if (i + 1) % 12 == 0:
            # Calculate cumulative inflation for the past year
            year_inflation = (1 + inflation_monthly.iloc[max(0, i-11):i+1]).prod() - 1
            # Adjust withdrawal for inflation annually (every 12 months)
            if (i + 1) % 12 == 0:
                year_inflation = (1 + inflation_monthly.iloc[max(0, i-11):i+1]).prod() - 1
                monthly_withdraw *= (1 + year_inflation)  # update only the ongoing withdrawal


    # Summary metrics
    final_balance = balances[-1]
    total_real_withdrawn = cum_withdrawals[-1]
    success = final_balance > 0
    
    # Display key metrics at the top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        balance_delta = final_balance - initial_balance
        delta_color = "normal" if balance_delta >= 0 else "inverse"
        st.metric("Final Balance", f"${final_balance:,.0f}", 
                 delta=f"${balance_delta:,.0f}", delta_color=delta_color)
    with col2:
        st.metric("Total Withdrawn", f"${total_real_withdrawn:,.0f}")
    with col3:
        st.metric("Portfolio Survived", "✅ Yes" if success else "❌ No")
    with col4:
        years_lasted = len([b for b in balances if b > 0]) / 12
        st.metric("Years Lasted", f"{years_lasted:.1f}")
    
    # Prepare Plotly Figure
    fig = go.Figure()
    
    # Portfolio balance
    fig.add_trace(go.Scatter(
        x=simulation_dates,
        y=balances,
        mode='lines',
        name='Portfolio Balance',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='<b>Balance</b>: $%{y:,.0f}<br><b>Date</b>: %{x}<extra></extra>'
    ))
    
    # Cumulative withdrawals
    fig.add_trace(go.Scatter(
        x=simulation_dates,
        y=cum_withdrawals,
        mode='lines',
        name='Cumulative Withdrawals',
        line=dict(color='#2ca02c', width=3),
        hovertemplate='<b>Withdrawn</b>: $%{y:,.0f}<br><b>Date</b>: %{x}<extra></extra>'
    ))
    
    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="$0")
    
    fig.update_layout(
        title=f"Portfolio Balance vs Cumulative Withdrawals ({withdraw_rate}% Rule - {country} Inflation)<br><sub>Period: {start_year} to {end_year}</sub>",
        xaxis_title="Date",
        yaxis=dict(
            title="USD",
            tickformat="$,.0f"
        ),
        template="plotly_white",
        height=600,
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    # Table Data (show annual snapshots)
    annual_indices = list(range(11, n_months, 12))  # Every 12th month
    if annual_indices and annual_indices[-1] != n_months - 1:
        annual_indices.append(n_months - 1)  # Add final month
    
    if annual_indices:
        table_df = pd.DataFrame({
            'Year': [simulation_dates[i].strftime('%Y') for i in annual_indices],
            'Date': [simulation_dates[i].strftime('%Y-%m') for i in annual_indices],
            'Portfolio Balance': [f"${balances[i]:,.0f}" for i in annual_indices],
            'Cumulative Withdrawn': [f"${cum_withdrawals[i]:,.0f}" for i in annual_indices],
            'Monthly Withdrawal': [f"${monthly_withdrawals[i]:,.0f}" for i in annual_indices]
        })
    else:
        # Fallback for very short simulations
        table_df = pd.DataFrame({
            'Year': [simulation_dates[-1].strftime('%Y')],
            'Date': [simulation_dates[-1].strftime('%Y-%m')],
            'Portfolio Balance': [f"${balances[-1]:,.0f}"],
            'Cumulative Withdrawn': [f"${cum_withdrawals[-1]:,.0f}"],
            'Monthly Withdrawal': [f"${monthly_withdrawals[-1]:,.0f}"]
        })
    
    # Layout: chart 70%, table 30%
    col1, col2 = st.columns([2.3, 1])
    with col1:
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.subheader("Annual Summary")
        st.dataframe(table_df, height=600, width='stretch', hide_index=True)
    
    # Additional analysis
    st.subheader("Historical Context")
    
    # Calculate some interesting statistics
    avg_annual_return = (growth_monthly.mean() * 12) * 100
    avg_annual_inflation = (inflation_monthly.mean() * 12) * 100
    real_return = avg_annual_return - avg_annual_inflation
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Annual Return", f"{avg_annual_return:.1f}%")
    with col2:
        st.metric("Avg Annual Inflation", f"{avg_annual_inflation:.1f}%")
    with col3:
        st.metric("Real Return", f"{real_return:.1f}%")
    
    # Assumptions and Calculations section
    st.subheader("Assumptions/Calculations")
    
    st.write(f"""
    **Data Sources and Values Used:**
    - **Stock Market Data**: Yahoo Finance {sp500_ticker} (S&P 500 {'Total Return Index' if 'TR' in sp500_ticker else 'Price Index'})
      - Uses {'Adjusted Close prices' if 'TR' in sp500_ticker else 'Close prices'} which include dividend reinvestment for total return calculation
      - Monthly returns calculated as: (Current Month Price / Previous Month Price) - 1
    
    - **Inflation Data**: Federal Reserve Economic Data (FRED) - {country} Consumer Price Index
      - Uses official government CPI data: {'CPIAUCSL (US CPI for All Urban Consumers)' if country == 'US' else 'CANCPIALLMINMEI (Canada CPI All Items)'}
      - Monthly inflation calculated as: (Current Month CPI / Previous Month CPI) - 1
    
    **Withdrawal Calculation Method:**
    - **Initial Monthly Withdrawal**: \${base_initial_monthly_withdraw:,.0f} = \${initial_balance:,} × {withdraw_rate}% ÷ 12 months
    - **Annual Inflation Adjustment**: Each year, the monthly withdrawal is increased by the actual cumulative inflation from the previous 12 months
    - **Compounding Effect**: Inflation adjustments compound over time, so a withdrawal that starts at \${base_initial_monthly_withdraw:,.0f} grows with actual historical inflation

    
    **Portfolio Mechanics:**
    - Each month: Portfolio grows by S&P 500 monthly return, then monthly withdrawal is subtracted
    - **Rebalancing**: Assumes perfect monthly rebalancing (no transaction costs)
    - **No Taxes**: Does not account for capital gains taxes or dividend taxes
    - **No Fees**: Does not include investment management fees or expense ratios
    """)
    
    # Data range explanation
    if df.index[0].year > data_start_date.year:
        st.info(f"""
        **📅 Data Range Note**: While CPI data is available from {data_start_date.strftime('%Y')}, the simulation can only start from 
        {df.index[0].strftime('%Y-%m')} because this is when both S&P 500 and CPI data have sufficient overlap and quality. 
        Early data may have gaps or inconsistencies that affect simulation reliability.
        """)
    
    # Simulation details
    st.subheader("Simulation Parameters")
    withdrawal_increase = ((monthly_withdrawals[-1] / monthly_withdrawals[0]) - 1) * 100
    
    st.write(f"""
    **Data Sources:**
    - **Stock Data**: Yahoo Finance ({sp500_ticker})
    - **Inflation Data**: Federal Reserve Economic Data (FRED) - {country} CPI
    - **Actual Data Range Used**: {simulation_dates[0].strftime('%Y-%m-%d')} to {simulation_dates[-1].strftime('%Y-%m-%d')}
    
    **Selected Simulation Period:**
    - **Period**: {simulation_dates[0].strftime('%B %Y')} to {simulation_dates[-1].strftime('%B %Y')}
    - **Total Duration**: {years} years ({n_months} months)
    
    **Results:**
    - **Initial Monthly Withdrawal**: ${base_initial_monthly_withdraw:,.0f}
    - **Final Monthly Withdrawal**: ${monthly_withdrawals[-1]:,.0f} (+{withdrawal_increase:.1f}% due to inflation)
    
    **Key Insights:**
    - Your withdrawal amount increased by {withdrawal_increase:.1f}% over {years} years due to inflation adjustment
    - The portfolio {'survived' if success else 'was depleted during'} the simulation period
    - Average real return (after inflation) was {real_return:.1f}% annually
    """)
    
    # Risk analysis
    if success:
        min_balance = min(balances)
        min_date = simulation_dates[balances.index(min_balance)]
        st.success(f"✅ **Portfolio Success**: The portfolio survived the full {years}-year period with ${final_balance:,.0f} remaining.")
        st.info(f"⚠️ **Lowest Point**: ${min_balance:,.0f} in {min_date.strftime('%B %Y')}")
    else:
        depletion_month = next((i for i, b in enumerate(balances) if b <= 0), len(balances))
        depletion_date = simulation_dates[min(depletion_month, len(simulation_dates)-1)]
        st.error(f"❌ **Portfolio Depletion**: The portfolio was depleted in {depletion_date.strftime('%B %Y')} after {depletion_month/12:.1f} years.")

else:
    st.error("Unable to load data from external sources. Please check your internet connection and try again.")
    st.info("""
    **Troubleshooting:**
    1. Check your internet connection
    2. Ensure you have the required packages:
    ```
    pip install yfinance requests pandas plotly streamlit numpy
    ```
    3. Try refreshing the page to retry data fetching
    """)

import matplotlib.pyplot as plt

st.write(f"""
**Raw market and inflation data used in the simulation**
""")
# 1x2 subplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('black')  # Set figure background to black

# ---------- S&P 500 subplot ----------
axes[0].set_facecolor('black')      # Set axes background to black
df['SP500'].plot(ax=axes[0], color='cyan', logy=True)  # semilog y, colorful line
axes[0].set_title("S&P 500 (^GSPC)", color='white', fontsize=14)
axes[0].set_xlabel("Year", color='white')
axes[0].set_ylabel("S&P 500 (log-scale)", color='white')
axes[0].tick_params(axis='x', colors='white')
axes[0].tick_params(axis='y', colors='white')

# ---------- Inflation subplot ----------
axes[1].set_facecolor('black')
df['US_CPI'].plot(ax=axes[1], color='orange', label="US CPI")
df['CAN_CPI'].plot(ax=axes[1], color='green', label="Canada CPI")
axes[1].set_title("Inflation (CPI)", color='white', fontsize=14)
axes[1].set_xlabel("Year", color='white')
axes[1].set_ylabel("CPI Index", color='white')
axes[1].legend(facecolor='black', edgecolor='white', labelcolor='white')
axes[1].tick_params(axis='x', colors='white')
axes[1].tick_params(axis='y', colors='white')

plt.tight_layout()
plt.show()

st.pyplot(fig)

