"""
VMarket - Volatility Prediction Platform
TradingView-style dashboard with AI-powered volatility forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from volatility_calculator import VolatilityCalculator, prepare_stock_data
from garch_model import VolatilityForecaster
from lstm_model import LSTMVolatilityPredictor
import torch

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="VMarket | AI Volatility Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Dark theme colors */
    :root {
        --bg-primary: #0E1117;
        --bg-secondary: #1c1f26;
        --accent-green: #00ff88;
        --accent-red: #ff4444;
        --accent-blue: #00a8ff;
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
    }
    
    /* Main app styling */
    .stApp {
        background-color: var(--bg-primary);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1c1f26 0%, #2a2d35 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11141a 0%, #1c1f26 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Headers */
    h1 {
        color: var(--accent-blue);
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    h2, h3 {
        color: var(--text-primary);
        font-weight: 700;
    }
    
    /* SelectBox styling */
    .stSelectbox > div > div {
        background-color: var(--bg-secondary);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, var(--accent-blue) 0%, #0077cc 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,168,255,0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--bg-secondary);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 600;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--accent-blue) 0%, #0077cc 100%);
        color: white;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
    }
    
    /* Custom alert boxes */
    .alert-box {
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: 500;
    }
    
    .alert-success {
        background: rgba(0,255,136,0.1);
        border-left: 4px solid var(--accent-green);
        color: var(--accent-green);
    }
    
    .alert-warning {
        background: rgba(255,136,0,0.1);
        border-left: 4px solid #ff8800;
        color: #ff8800;
    }
    
    .alert-danger {
        background: rgba(255,68,68,0.1);
        border-left: 4px solid var(--accent-red);
        color: var(--accent-red);
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================
COMPANIES = {
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology'},
    'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology'},
    'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology'},
    'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology'},
    'META': {'name': 'Meta Platforms Inc.', 'sector': 'Technology'},
    'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer Cyclical'},
    'NFLX': {'name': 'Netflix Inc.', 'sector': 'Communication Services'}
}

# ==================== HELPER FUNCTIONS ====================
@st.cache_data
def load_stock_data(ticker):
    """Load and process stock data with caching"""
    file_path = f"{ticker}_stock_data.csv"
    if not os.path.exists(file_path):
        return None, None
    
    _, processed_df = prepare_stock_data(file_path)
    return processed_df

@st.cache_data
def load_sentiment_data(ticker):
    """Load sentiment data if available"""
    file_path = f"{ticker}_daily_sentiment.csv"
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

@st.cache_resource
def load_lstm_model(ticker):
    """Load trained LSTM model if available"""
    model_path = f"models/{ticker}_lstm.pth"
    if not os.path.exists(model_path):
        return None
    
    try:
        predictor = LSTMVolatilityPredictor(sequence_length=60, hidden_dim=128, num_layers=2)
        predictor.load_model(model_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading LSTM model: {str(e)}")
        return None

@st.cache_data
def load_training_data(ticker):
    """Load merged training data with sentiment"""
    file_path = f"{ticker}_training_data.csv"
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

def create_candlestick_chart(df, ticker, show_volume=True):
    """Create TradingView-style candlestick chart"""
    # Determine if we need subplots for volume
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.03,
            subplot_titles=('Price', 'Volume'),
            shared_xaxes=True
        )
    else:
        fig = go.Figure()
    
    # Candlestick
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444',
        increasing_fillcolor='rgba(0,255,136,0.3)',
        decreasing_fillcolor='rgba(255,68,68,0.3)'
    )
    
    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
        
        # Volume bars
        colors = ['#00ff88' if close >= open else '#ff4444' 
                  for close, open in zip(df['Close'], df['Open'])]
        
        volume = go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker=dict(color=colors, opacity=0.6),
            showlegend=False
        )
        fig.add_trace(volume, row=2, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"{ticker} - {COMPANIES[ticker]['name']}",
            font=dict(size=24, color='#00a8ff', family='Arial Black')
        ),
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1c1f26',
        height=600,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            gridwidth=1,
            title='Price ($)'
        )
    )
    
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_volatility_chart(df):
    """Create multi-volatility comparison chart"""
    fig = go.Figure()
    
    # Plot different volatility measures
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Vol_Close'],
        name='Close-to-Close',
        line=dict(color='#00a8ff', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Vol_GK'],
        name='Garman-Klass',
        line=dict(color='#00ff88', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Vol_30D'],
        name='30-Day Average',
        line=dict(color='#ff8800', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Historical Volatility Analysis',
        xaxis_title='Date',
        yaxis_title='Annualized Volatility',
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1c1f26',
        height=400,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(28,31,38,0.8)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        )
    )
    
    return fig

def create_volatility_heatmap(tickers_data):
    """Create volatility heat map for multiple stocks"""
    # Prepare data
    data = []
    for ticker, df in tickers_data.items():
        if df is not None and 'Vol_GK' in df.columns:
            recent_vol = df['Vol_GK'].iloc[-1]
            avg_vol = df['Vol_GK'].mean()
            data.append({
                'Ticker': ticker,
                'Company': COMPANIES[ticker]['name'],
                'Current Vol': recent_vol,
                'Average Vol': avg_vol,
                'Vol Change': ((recent_vol - avg_vol) / avg_vol) * 100
            })
    
    if not data:
        return None
    
    heat_df = pd.DataFrame(data)
    
    fig = go.Figure(data=go.Heatmap(
        z=heat_df[['Current Vol']].values.T,
        x=heat_df['Ticker'],
        y=['Volatility'],
        colorscale=[
            [0, '#00ff88'],
            [0.5, '#ffaa00'],
            [1, '#ff4444']
        ],
        text=heat_df[['Current Vol']].values.T,
        texttemplate='%{text:.1%}',
        textfont={"size": 14, "color": "white"},
        showscale=True,
        colorbar=dict(title="Volatility")
    ))
    
    fig.update_layout(
        title='Current Volatility Heat Map',
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1c1f26',
        height=200,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    
    return fig

def create_sentiment_chart(sentiment_df):
    """Create sentiment timeline chart"""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        subplot_titles=('Sentiment Score', 'News Volume'),
        shared_xaxes=True
    )
    
    # Sentiment line
    fig.add_trace(go.Scatter(
        x=sentiment_df.index,
        y=sentiment_df['sentiment_mean'],
        name='Sentiment',
        line=dict(color='#00a8ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,168,255,0.2)'
    ), row=1, col=1)
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#666", row=1, col=1)
    
    # News volume bars
    colors = ['#00ff88' if s >= 0 else '#ff4444' for s in sentiment_df['sentiment_mean']]
    fig.add_trace(go.Bar(
        x=sentiment_df.index,
        y=sentiment_df['news_count'],
        name='News Count',
        marker=dict(color=colors, opacity=0.6)
    ), row=2, col=1)
    
    fig.update_layout(
        title='News Sentiment Timeline',
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1c1f26',
        height=500,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode='x unified',
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Sentiment", row=1, col=1)
    fig.update_yaxes(title_text="Articles", row=2, col=1)
    
    return fig

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>
        📈 VMarket
    </h1>
    <p style='text-align: center; color: #a0a0a0; font-size: 1.2rem; margin-top: 0;'>
        AI-Powered Volatility Prediction Terminal
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("### 🎯 Analysis Controls")
        
        # Stock selector
        selected_ticker = st.selectbox(
            "Select Stock",
            options=list(COMPANIES.keys()),
            format_func=lambda x: f"{x} - {COMPANIES[x]['name']}"
        )
        
        st.markdown("---")
        
        # Date range
        st.markdown("### 📅 Time Period")
        time_period = st.selectbox(
            "Select Range",
            options=['1M', '3M', '6M', '1Y', '2Y', 'MAX'],
            index=4
        )
        
        st.markdown("---")
        
        # Analysis options
        st.markdown("### ⚙️ Display Options")
        show_volume = st.checkbox("Show Volume", value=True)
        show_predictions = st.checkbox("Show Predictions", value=True)
        show_sentiment = st.checkbox("Show Sentiment", value=True)
        
        st.markdown("---")
        
        # Info box
        st.markdown("""
        <div class="alert-box alert-success">
            <strong>✓ Models Active</strong><br>
            • GARCH(1,1) Baseline<br>
            • LSTM Deep Learning<br>
            • Sentiment Analysis
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== MAIN CONTENT ====================
    
    # Load data
    df = load_stock_data(selected_ticker)
    
    if df is None:
        st.error(f"❌ No data found for {selected_ticker}")
        return
    
    # Filter by time period
    period_map = {
        '1M': 30, '3M': 90, '6M': 180,
        '1Y': 365, '2Y': 730, 'MAX': len(df)
    }
    days = period_map.get(time_period, 365)
    df_filtered = df.iloc[-days:]
    
    # ==== TOP METRICS ====
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = df_filtered['Close'].iloc[-1]
    price_change = df_filtered['Close'].iloc[-1] - df_filtered['Close'].iloc[-2]
    price_change_pct = (price_change / df_filtered['Close'].iloc[-2]) * 100
    
    current_vol = df_filtered['Vol_GK'].iloc[-1] if 'Vol_GK' in df_filtered.columns else 0
    avg_vol = df_filtered['Vol_GK'].mean() if 'Vol_GK' in df_filtered.columns else 0
    
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        st.metric(
            "Current Volatility",
            f"{current_vol:.1%}",
            f"{((current_vol - avg_vol) / avg_vol * 100):+.1f}% vs Avg"
        )
    
    with col3:
        st.metric(
            "30-Day Avg Vol",
            f"{avg_vol:.1%}"
        )
    
    with col4:
        high_52w = df_filtered['High'].max()
        st.metric(
            "52W High",
            f"${high_52w:.2f}"
        )
    
    with col5:
        low_52w = df_filtered['Low'].min()
        st.metric(
            "52W Low",
            f"${low_52w:.2f}"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==== MAIN CHART ====
    chart_tab, vol_tab, pred_tab, data_tab = st.tabs([
        "📊 Price Chart",
        "📈 Volatility Analysis",
        "🤖 AI Predictions",
        "📋 Data Table"
    ])
    
    with chart_tab:
        fig_candlestick = create_candlestick_chart(df_filtered, selected_ticker, show_volume)
        st.plotly_chart(fig_candlestick, use_container_width=True)
    
    with vol_tab:
        if 'Vol_GK' in df_filtered.columns:
            fig_vol = create_volatility_chart(df_filtered)
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Volatility statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min Volatility", f"{df_filtered['Vol_GK'].min():.2%}")
            with col2:
                st.metric("Max Volatility", f"{df_filtered['Vol_GK'].max():.2%}")
            with col3:
                st.metric("Std Dev", f"{df_filtered['Vol_GK'].std():.2%}")
        else:
            st.warning("Volatility data not available")
    
    with pred_tab:
        # Load models and data
        lstm_model = load_lstm_model(selected_ticker)
        training_data = load_training_data(selected_ticker)
        sentiment_data = load_sentiment_data(selected_ticker)
        
        # Create tabs for different models
        garch_tab, lstm_tab, ensemble_tab, sentiment_tab = st.tabs([
            "📊 GARCH", "🧠 LSTM", "🎯 Ensemble", "📰 Sentiment"
        ])
        
        # ===== GARCH TAB =====
        with garch_tab:
            st.markdown("### GARCH(1,1) Volatility Forecast")
            
            try:
                with st.spinner("Training GARCH model..."):
                    forecaster = VolatilityForecaster()
                    stock_df = pd.DataFrame(df_filtered)
                    forecaster.train_model(selected_ticker, stock_df)
                    prediction = forecaster.predict_volatility(selected_ticker, horizon=1)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Volatility", f"{prediction['predicted_volatility']:.2%}")
                with col2:
                    st.metric("Lower Bound (95%)", f"{prediction['lower_bound']:.2%}")
                with col3:
                    st.metric("Upper Bound (95%)", f"{prediction['upper_bound']:.2%}")
                
                hist_vol = forecaster.get_historical_fit(selected_ticker)
                
                fig_garch = go.Figure()
                fig_garch.add_trace(go.Scatter(
                    x=df_filtered.index[-len(hist_vol):],
                    y=hist_vol,
                    name='GARCH Fitted',
                    line=dict(color='#00a8ff', width=2)
                ))
                fig_garch.add_trace(go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered['Vol_GK'],
                    name='Realized',
                    line=dict(color='#00ff88', width=2, dash='dot')
                ))
                
                fig_garch.update_layout(
                    title='GARCH Model Fit',
                    template='plotly_dark',
                    paper_bgcolor='#0E1117',
                    plot_bgcolor='#1c1f26',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_garch, use_container_width=True)
                
            except Exception as e:
                st.error(f"GARCH Error: {str(e)}")
        
        # ===== LSTM TAB =====
        with lstm_tab:
            st.markdown("### LSTM Deep Learning Forecast")
            
            if lstm_model is not None and training_data is not None:
                try:
                    # Prepare last sequence for prediction
                    X, _ = lstm_model.prepare_data(training_data.iloc[-100:], target_col='Vol_GK')
                    
                    if len(X) > 0:
                        lstm_pred = lstm_model.predict(X[-1:])
                        lstm_vol = lstm_pred[0]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("LSTM Predicted Volatility", f"{lstm_vol:.2%}")
                        with col2:
                            diff = ((lstm_vol - current_vol) / current_vol * 100)
                            st.metric("vs Current", f"{diff:+.1f}%")
                        
                        # Show model info
                        st.info(f"✓ Using GPU-trained model ({len(X)} sequences, 10 features)")
                    else:
                        st.warning("Insufficient data for LSTM prediction")
                        
                except Exception as e:
                    st.error(f"LSTM Error: {str(e)}")
            else:
                st.warning("LSTM model not available for this stock")
        
        # ===== ENSEMBLE TAB =====
        with ensemble_tab:
            st.markdown("### 🎯 Ensemble Prediction (GARCH + LSTM)")
            
            try:
                # Get GARCH prediction
                forecaster = VolatilityForecaster()
                stock_df = pd.DataFrame(df_filtered)
                forecaster.train_model(selected_ticker, stock_df)
                garch_pred = forecaster.predict_volatility(selected_ticker, horizon=1)
                
                # Get LSTM prediction if available
                if lstm_model is not None and training_data is not None:
                    X, _ = lstm_model.prepare_data(training_data.iloc[-100:], target_col='Vol_GK')
                    if len(X) > 0:
                        lstm_vol = lstm_model.predict(X[-1:])[0]
                        
                        # Weighted ensemble (60% LSTM, 40% GARCH)
                        ensemble_vol = 0.6 * lstm_vol + 0.4 * garch_pred['predicted_volatility']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("GARCH", f"{garch_pred['predicted_volatility']:.2%}")
                        with col2:
                            st.metric("LSTM", f"{lstm_vol:.2%}")
                        with col3:
                            st.metric("Ensemble", f"{ensemble_vol:.2%}", delta=f"{((ensemble_vol - current_vol) / current_vol * 100):+.1f}%")
                        with col4:
                            st.metric("Current", f"{current_vol:.2%}")
                        
                        # Alert
                        if ensemble_vol > current_vol * 1.2:
                            st.markdown('<div class="alert-box alert-danger">⚠️ <strong>High Volatility Alert!</strong></div>', unsafe_allow_html=True)
                        elif ensemble_vol < current_vol * 0.8:
                            st.markdown('<div class="alert-box alert-success">✓ <strong>Low Volatility Expected</strong></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="alert-box alert-warning">→ <strong>Stable Volatility</strong></div>', unsafe_allow_html=True)
                    else:
                        st.info("Using GARCH only (insufficient LSTM data)")
                        st.metric("Predicted Volatility", f"{garch_pred['predicted_volatility']:.2%}")
                else:
                    st.info("Using GARCH only (LSTM not available)")
                    st.metric("Predicted Volatility", f"{garch_pred['predicted_volatility']:.2%}")
                    
            except Exception as e:
                st.error(f"Ensemble Error: {str(e)}")
        
        # ===== SENTIMENT TAB =====
        with sentiment_tab:
            if sentiment_data is not None and show_sentiment:
                st.markdown("### 📰 News Sentiment Analysis")
                
                # Display sentiment chart
                fig_sent = create_sentiment_chart(sentiment_data)
                st.plotly_chart(fig_sent, use_container_width=True)
                
                # Sentiment metrics
                col1, col2, col_3 = st.columns(3)
                with col1:
                    avg_sent = sentiment_data['sentiment_mean'].mean()
                    st.metric("Avg Sentiment (30d)", f"{avg_sent:.3f}")
                with col2:
                    latest_sent = sentiment_data['sentiment_mean'].iloc[-1]
                    st.metric("Latest Sentiment", f"{latest_sent:.3f}")
                with col_3:
                    total_news = sentiment_data['news_count'].sum()
                    st.metric("Total Articles", f"{int(total_news)}")
                    
            else:
                st.info("No sentiment data available for this stock")
    
    with data_tab:
        st.markdown("### 📊 Recent Data")
        display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Vol_GK', 'Intraday_Range']
        available_cols = [col for col in display_cols if col in df_filtered.columns]
        st.dataframe(
            df_filtered[available_cols].tail(50).style.format({
                col: '{:.2f}' if col != 'Volume' else '{:,.0f}'
                for col in available_cols
            }),
            use_container_width=True
        )
    
    # ==================== MULTI-STOCK COMPARISON ====================
    st.markdown("---")
    st.markdown("## 🌐 Market Overview")
    
    # Load all stock data
    all_data = {}
    for ticker in COMPANIES.keys():
        data = load_stock_data(ticker)
        if data is not None:
            all_data[ticker] = data
    
    if len(all_data) > 1:
        heat_fig = create_volatility_heatmap(all_data)
        if heat_fig:
            st.plotly_chart(heat_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>⚠️ <strong>Disclaimer:</strong> This is an educational project. Not financial advice. Do not invest real money based on these predictions.</p>
        <p style='font-size: 0.9rem;'>Built with Streamlit • GARCH • FinBERT • LSTM</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
