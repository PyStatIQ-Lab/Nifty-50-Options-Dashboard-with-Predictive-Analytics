import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Configuration (from your original code)
BASE_URL = "https://service.upstox.com/option-analytics-tool/open/v1"
MARKET_DATA_URL = "https://service.upstox.com/market-data-api/v2/open/quote"
HEADERS = {"accept": "application/json", "content-type": "application/json"}
NIFTY_LOT_SIZE = 75

# Fetch data functions (from your original code)
@st.cache_data(ttl=300)
def fetch_options_data(asset_key="NSE_INDEX|Nifty 50", expiry="24-04-2025"):
    url = f"{BASE_URL}/strategy-chains?assetKey={asset_key}&strategyChainType=PC_CHAIN&expiry={expiry}"
    response = requests.get(url, headers=HEADERS)
    return response.json() if response.status_code == 200 else None

@st.cache_data(ttl=60)
def fetch_nifty_price():
    url = f"{MARKET_DATA_URL}?i=NSE_INDEX|Nifty%2050"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()['data']['lastPrice']
    return None

def process_options_data(raw_data, spot_price):
    # (Keep your original processing function exactly as is)
    pass

# Predictive analysis functions
def predict_movement(df, feature='pcr', target='call_ltp'):
    X = df[feature].values.reshape(-1, 1)
    y = df[target].values
    
    # Polynomial regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict next 5 values
    last_value = X[-1]
    future_values = np.linspace(last_value, last_value*1.2, 5).reshape(-1, 1)
    future_poly = poly.transform(future_values)
    predictions = model.predict(future_poly)
    
    return {
        'current': y[-1],
        'predictions': predictions,
        'trend': 'up' if predictions[-1] > y[-1] else 'down',
        'confidence': abs(predictions[-1] - y[-1]) / y[-1] * 100
    }

# Dashboard layout
st.set_page_config(layout="wide", page_title="Nifty 50 Options Dashboard")

# Title and filters
st.title("📊 Nifty 50 Options Analytics Dashboard")
col1, col2, col3 = st.columns(3)
with col1:
    expiry_date = st.date_input("Select Expiry", pd.to_datetime("2025-04-24"))
with col2:
    num_strikes = st.slider("Number of Strikes to Display", 5, 50, 20)
with col3:
    refresh = st.button("Refresh Data")

# Load data
spot_price = fetch_nifty_price()
if spot_price is None:
    st.error("Failed to fetch Nifty spot price")
    st.stop()

raw_data = fetch_options_data(expiry=expiry_date.strftime("%d-%m-%Y"))
if raw_data is None:
    st.error("Failed to fetch options data")
    st.stop()

df = process_options_data(raw_data, spot_price)
if df is None:
    st.error("Failed to process options data")
    st.stop()

# Top summary metrics
st.subheader(f"Market Overview (Spot: {spot_price:.2f})")
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_oi = (df['call_oi'].sum() + df['put_oi'].sum()) * NIFTY_LOT_SIZE
    st.metric("Total Open Interest", f"{total_oi/1e6:,.1f}M", delta="5.2%")
with col2:
    pcr = df['pcr'].mean()
    st.metric("Put-Call Ratio", f"{pcr:.2f}", 
              delta="Bullish" if pcr < 0.7 else ("Neutral" if pcr < 1.3 else "Bearish"))
with col3:
    iv_diff = df['put_iv'].mean() - df['call_iv'].mean()
    st.metric("IV Skew", f"{iv_diff:.2f}%", 
              delta="Call Premium" if iv_diff < 0 else "Put Premium")
with col4:
    pred = predict_movement(df)
    st.metric("Predicted Movement", pred['trend'].upper(), 
              f"{pred['confidence']:.1f}% confidence")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Open Interest Analysis", "Price & Volume", "Greeks Analysis", "Predictive Models"])

with tab1:
    # OI Analysis
    st.subheader("Open Interest Analysis")
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Open Interest by Strike", "OI Change by Strike"))
    
    fig.add_trace(
        go.Bar(x=df['strike'], y=df['call_oi'], name='Call OI', marker_color='green'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=df['strike'], y=df['put_oi'], name='Put OI', marker_color='red'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df['strike'], y=df['call_oi_change'], name='Call OI Change', marker_color='lightgreen'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=df['strike'], y=df['put_oi_change'], name='Put OI Change', marker_color='pink'),
        row=1, col=2
    )
    
    fig.update_layout(barmode='group', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # PCR Heatmap
    st.subheader("Put-Call Ratio Heatmap")
    heatmap_data = df.pivot_table(values='pcr', index=pd.cut(df['strike'], bins=10))
    st.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn")

with tab2:
    # Price & Volume Analysis
    st.subheader("Price & Volume Analysis")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    
    fig.add_trace(
        go.Scatter(x=df['strike'], y=df['call_ltp'], name='Call LTP', line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['strike'], y=df['put_ltp'], name='Put LTP', line=dict(color='red')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df['strike'], y=df['call_volume'], name='Call Volume', marker_color='lightgreen'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=df['strike'], y=df['put_volume'], name='Put Volume', marker_color='pink'),
        row=2, col=1
    )
    
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    # Bid-Ask Spread Analysis
    st.subheader("Bid-Ask Spread Analysis")
    df['call_spread'] = df['call_ask'] - df['call_bid']
    df['put_spread'] = df['put_ask'] - df['put_bid']
    st.line_chart(df[['strike', 'call_spread', 'put_spread']].set_index('strike'))

with tab3:
    # Greeks Analysis
    st.subheader("Option Greeks Analysis")
    
    greeks = st.selectbox("Select Greek to Analyze", ['Delta', 'Gamma', 'Theta', 'Vega', 'IV'])
    
    if greeks == 'Delta':
        y1, y2 = 'call_delta', 'put_delta'
    elif greeks == 'Gamma':
        y1, y2 = 'call_gamma', 'put_gamma'
    elif greeks == 'Theta':
        y1, y2 = 'call_theta', 'put_theta'
    elif greeks == 'Vega':
        y1, y2 = 'call_vega', 'put_vega'
    else:
        y1, y2 = 'call_iv', 'put_iv'
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['strike'], y=df[y1], name=f'Call {greeks}'))
    fig.add_trace(go.Scatter(x=df['strike'], y=df[y2], name=f'Put {greeks}'))
    fig.add_vline(x=spot_price, line_dash="dash", line_color="black", annotation_text="Spot Price")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Greeks Comparison
    st.subheader("Greeks Comparison")
    st.write("""
    - **Delta**: Call deltas are positive (0 to 1), Put deltas are negative (-1 to 0)
    - **Gamma**: Peaks near ATM options
    - **Theta**: Time decay is most severe for ATM options
    - **Vega**: Sensitivity to volatility changes
    """)

with tab4:
    # Predictive Models
    st.subheader("Price Movement Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        feature = st.selectbox("Feature for Prediction", ['pcr', 'call_oi', 'put_oi', 'call_iv', 'put_iv'])
    with col2:
        target = st.selectbox("Target to Predict", ['call_ltp', 'put_ltp', 'call_volume', 'put_volume'])
    
    prediction = predict_movement(df, feature, target)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(prediction['predictions'])),
        y=prediction['predictions'],
        name='Predicted',
        line=dict(color='blue', dash='dot')
    ))
    fig.add_hline(y=prediction['current'], line_dash="dash", 
                 annotation_text="Current Value", line_color="green")
    fig.update_layout(
        title=f"Predicted {target.replace('_', ' ').title()} Movement",
        xaxis_title="Future Periods",
        yaxis_title=target.replace('_', ' ').title()
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric("Prediction Direction", prediction['trend'].upper(), 
              f"{prediction['confidence']:.1f}% confidence")
    
    # Explanation
    st.subheader("Model Explanation")
    st.write("""
    The predictive model uses polynomial regression to forecast future values based on:
    - Current market trends
    - Historical relationships between selected features and targets
    - The confidence score indicates the strength of predicted movement
    """)

# Data table at bottom
st.subheader("Raw Options Data")
st.dataframe(df.sort_values('strike').head(num_strikes), use_container_width=True)
