import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Core Simulation ---
def retirement_simulation(inflation_rate=4.0, growth_rate=10.0, years=30, 
                          initial_balance=1000000, withdraw_rate=4.0):
    months = years * 12
    balance = initial_balance
    withdrawal = initial_balance * withdraw_rate / 100 / 12
    balances = []
    monthly_withdrawals = []
    annual_withdrawals = []
    yearly_sum = 0
    depletion_month = None

    for m in range(1, months + 1):
        balance *= (1 + growth_rate / 100 / 12)
        balance -= withdrawal
        balances.append(balance)
        monthly_withdrawals.append(withdrawal)
        yearly_sum += withdrawal

        if depletion_month is None and balance <= 0:
            depletion_month = m

        if m % 12 == 0:
            annual_withdrawals.append(yearly_sum)
            yearly_sum = 0
            withdrawal *= (1 + inflation_rate / 100)

    cumulative_withdrawals = np.cumsum(annual_withdrawals).tolist()
    return np.arange(1, months + 1), balances, monthly_withdrawals, annual_withdrawals, cumulative_withdrawals, depletion_month

# --- Split segments for green/red shading ---
def split_segments(x, y1, y2):
    segments = []
    current_x, current_y1, current_y2, current_color = [], [], [], None
    for xi, b, c in zip(x, y1, y2):
        color = 'green' if b >= c else 'red'
        if current_color is None:
            current_color = color
        if color != current_color:
            if current_x:
                segments.append((current_x, current_y1, current_y2, current_color))
            current_x, current_y1, current_y2 = [], [], []
            current_color = color
        current_x.append(xi)
        current_y1.append(b)
        current_y2.append(c)
    if current_x:
        segments.append((current_x, current_y1, current_y2, current_color))
    return segments

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Retirement Portfolio Simulation (4% Rule)")
# Subtitle / explanation text
st.subheader("About this tool")
st.markdown(
    """
    The 4% rule is a retirement guideline that says:
    - You can withdraw 4% of your initial retirement portfolio each year, adjusted for inflation,
    - and your money should last about 30 years (historically, in U.S. markets).
    
    #### Example:
    - If you retire with \$1,000,000, you withdraw \$40,000 in year 1.
    - In year 2, you increase the withdrawal by inflation (say 3%), so you withdraw $41,200, etc.
    - Meanwhile, your portfolio is (hopefully) growing at some average annual rate.  
    #### Your Dynamic Simulator has:
    - Inputs: average inflation rate (%) and average annual growth rate (%)
    - Starting balance = \$1,000,000
    - Withdrawals = 4% of initial balance (\$40,000 per year), split into 12 monthly withdrawals, adjusted yearly for inflation
    - Growth = applied monthly (annual growth / 12)
    #### Features:
    - Balance evolution over time with monthly withdrawals.
    - Inflation-adjusted withdrawals.
    - Sliders to adjust average inflation and growth rate.
    - Interactive Plotly graph (zoom, hover, etc).
    """
)

# Sliders in horizontal columns for compact layout
col1, col2, col3 = st.columns([1,1,1])
inflation = col1.slider("Inflation Rate (%)", 0.0, 10.0, 4.0, 0.5, format="%.1f")
growth = col2.slider("Average Annual Growth Rate", 0.0, 20.0, 5.0, 0.5, format="%.1f")
years  = col3.slider("Years", 10, 50, 30, 1)

# Run simulation
months, balances, monthly_withdrawals, annual_withdrawals, cumulative_withdrawals, depletion_month = retirement_simulation(
    inflation, growth, years
)
months_years = np.arange(1, years*12+1)/12
cumulative_months = np.arange(12, years*12+1, 12)
cumulative_full = np.interp(np.arange(1, years*12+1), cumulative_months, cumulative_withdrawals)

# --- Plotly Chart + Table ---
fig = make_subplots(
    rows=1, cols=2, column_widths=[0.65, 0.35],
    specs=[[{"secondary_y": True}, {"type": "table"}]],
    subplot_titles=("Portfolio Balance & Withdrawals", "Annual & Cumulative Withdrawals")
)

# Portfolio balance line
fig.add_trace(
    go.Scatter(x=months_years, y=balances, mode="lines", name="Portfolio Balance", line=dict(color="blue")),
    row=1, col=1, secondary_y=False
)

# Cumulative withdrawals line
fig.add_trace(
    go.Scatter(x=months_years, y=cumulative_full, mode="lines+markers",
               name="Cumulative Withdrawals", line=dict(dash='dot', color="red")),
    row=1, col=1, secondary_y=False
)

# Safety margin shading
segments = split_segments(months_years, balances, cumulative_full)
for seg_x, seg_y1, seg_y2, color in segments:
    fig.add_trace(
        go.Scatter(
            x=seg_x,
            y=seg_y1,
            fill=None,
            mode='lines',
            line=dict(color='blue'),
            showlegend=False
        ),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=seg_x,
            y=seg_y2,
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(255,0,0,0.0)'),
            fillcolor='rgba(0,200,0,0.2)' if color=='green' else 'rgba(255,0,0,0.2)',
            showlegend=False
        ),
        row=1, col=1, secondary_y=False
    )

# Monthly withdrawals
fig.add_trace(
    go.Scatter(x=months_years, y=monthly_withdrawals, mode="lines", name="Monthly Withdrawal",
               line=dict(color="orange", dash='dash')),
    row=1, col=1, secondary_y=True
)

# Depletion marker
if depletion_month is not None:
    fig.add_trace(
        go.Scatter(
            x=[depletion_month/12],
            y=[0],
            mode='markers+text',
            marker=dict(color='red', size=12, symbol='x'),
            text=["Portfolio Depleted"],
            textposition="top center",
            name="Depletion"
        ),
        row=1, col=1, secondary_y=False
    )

# Table in Plotly with integer formatting
years_list = list(range(1, len(annual_withdrawals)+1))
fig.add_trace(
    go.Table(
        header=dict(values=["Year", "Annual Withdrawals ($)", "Cumulative Withdrawals ($)"],
                    fill_color="#d9d9d9", font=dict(color="black", size=13), align="center"),
        cells=dict(values=[
            years_list,
            [f"{w:,.0f}" for w in annual_withdrawals],
            [f"{c:,.0f}" for c in cumulative_withdrawals]
        ], fill_color="#f2f2f2", font=dict(color="black", size=12), align="center", height=25)
    ),
    row=1, col=2
)

# Layout styling
fig.update_layout(
    width=1300, height=600,
    template="plotly",
    plot_bgcolor="rgba(240,240,240,1)",
    paper_bgcolor="rgba(245,245,245,1)",
    font=dict(color="black"),
    showlegend=True,
    legend=dict(
        x=0.02, y=0.98,  # top-left inside graph
        font=dict(color="black"),
        bordercolor="black", borderwidth=1
    ),
    margin=dict(l=50, r=50, t=50, b=50),
    title_font=dict(color="black", size=18)
)

# Axis titles and tick labels in black
fig.update_xaxes(title_text="Years", title_font=dict(color="black", size=14), tickfont=dict(color="black", size=12))
fig.update_yaxes(title_text="Balance / Cumulative Withdrawals ($)", secondary_y=False,
                 title_font=dict(color="black", size=14), tickfont=dict(color="black", size=12))
fig.update_yaxes(title_text="Monthly Withdrawal ($)", secondary_y=True,
                 title_font=dict(color="black", size=14), tickfont=dict(color="black", size=12))

# --- Display ---
st.plotly_chart(fig, use_container_width=True)
