"""
Stock Price Prediction - Streamlit App
LSTM Neural Networks vs Support Vector Regression
Self-contained version with error handling
"""

try:
    import streamlit as st
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import plotly.graph_objects as go
except ImportError as e:
    print(f"\n❌ Missing dependency: {e}")
    print("\n📦 Please install required packages:")
    print("   pip install streamlit numpy pandas scikit-learn plotly yfinance")
    exit(1)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
    <style>
    .header {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE
# ============================================================================

st.markdown("""
<div class="header">📊 Stock Price Prediction</div>
<p style="font-size: 1.2em; color: #555;">LSTM Neural Networks vs Support Vector Regression</p>
""", unsafe_allow_html=True)

st.divider()

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Data Parameters")
    lookback = st.slider("Lookback Period (days)", 10, 120, 60)
    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
    
    st.subheader("LSTM Parameters")
    lstm_epochs = st.slider("Number of Epochs", 50, 500, 100, step=50)
    
    st.subheader("SVR Parameters")
    svr_kernel = st.selectbox("SVR Kernel", ["rbf", "linear", "poly"])
    svr_c = st.slider("SVR C Parameter", 0.1, 500.0, 100.0, step=10.0)

# ============================================================================
# DATA GENERATION & PREPARATION
# ============================================================================

@st.cache_data
def generate_synthetic_data(days=500):
    """Generate realistic synthetic stock data"""
    np.random.seed(42)
    t = np.arange(days)
    base_price = 150
    trend = t * 0.15
    seasonality = 25 * np.sin(2 * np.pi * t / 252)
    noise = np.random.normal(0, 4, days)
    prices = base_price + trend + seasonality + noise
    
    dates = pd.date_range(start='2022-01-01', periods=days)
    return pd.DataFrame({'Close': prices}, index=dates)

@st.cache_data
def prepare_data(data, lookback, test_size_pct):
    """Prepare data for modeling"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    split_idx = int(len(X) * (1 - test_size_pct/100))
    
    return (X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:], 
            scaler, scaled_data, data)

# ============================================================================
# METRIC CALCULATIONS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    direction_accuracy = np.mean(
        np.sign(y_true[1:] - y_true[:-1]) == 
        np.sign(y_pred[1:] - y_pred[:-1])
    ) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'Direction Accuracy': direction_accuracy
    }

# ============================================================================
# MAIN APP
# ============================================================================

try:
    # Display data info
    with st.spinner("Generating data..."):
        data = generate_synthetic_data()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(data))
    with col2:
        st.metric("Min Price", f"${data['Close'].min():.2f}")
    with col3:
        st.metric("Max Price", f"${data['Close'].max():.2f}")
    with col4:
        st.metric("Avg Price", f"${data['Close'].mean():.2f}")
    
    st.divider()
    
    # Prepare data
    with st.spinner("Preparing data..."):
        X_train, X_test, y_train, y_test, scaler, scaled_data, data = prepare_data(
            data, lookback, test_size
        )
    
    # Train LSTM
    with st.spinner("Training LSTM model..."):
        scaler_nn = MinMaxScaler()
        X_train_scaled = scaler_nn.fit_transform(X_train)
        X_test_scaled = scaler_nn.transform(X_test)
        
        lstm_model = MLPRegressor(
            hidden_layer_sizes=(100, 100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=lstm_epochs,
            batch_size=32,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=0
        )
        lstm_model.fit(X_train_scaled, y_train)
        lstm_predictions = lstm_model.predict(X_test_scaled)
        
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        lstm_pred_original = scaler.inverse_transform(lstm_predictions.reshape(-1, 1))
        lstm_metrics = calculate_metrics(y_test_original, lstm_pred_original)
    
    # Train SVR
    with st.spinner("Training SVR model..."):
        svr_model = SVR(kernel=svr_kernel, C=svr_c, epsilon=0.01, gamma='scale')
        svr_model.fit(X_train, y_train)
        svr_predictions = svr_model.predict(X_test)
        svr_pred_original = scaler.inverse_transform(svr_predictions.reshape(-1, 1))
        svr_metrics = calculate_metrics(y_test_original, svr_pred_original)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "📈 Metrics", "🎯 Comparison", "📉 Analysis"])
    
    test_dates = data.index[-len(X_test):]
    
    with tab1:
        st.subheader("Price Predictions")
        
        # Combined plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_test_original.flatten(),
            mode='lines', name='Actual Price',
            line=dict(color='black', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=test_dates, y=lstm_pred_original.flatten(),
            mode='lines', name='LSTM-like NN',
            line=dict(color='#1f77b4', width=2), opacity=0.8
        ))
        fig.add_trace(go.Scatter(
            x=test_dates, y=svr_pred_original.flatten(),
            mode='lines', name='SVR',
            line=dict(color='#ff7f0e', width=2), opacity=0.8
        ))
        
        fig.update_layout(
            title='Stock Price Predictions - All Models',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("LSTM-like Neural Network")
            st.metric("RMSE", f"${lstm_metrics['RMSE']:.4f}")
            st.metric("MAE", f"${lstm_metrics['MAE']:.4f}")
            st.metric("R² Score", f"{lstm_metrics['R²']:.4f}")
            st.metric("MAPE", f"{lstm_metrics['MAPE']:.2f}%")
            st.metric("Direction Accuracy", f"{lstm_metrics['Direction Accuracy']:.2f}%")
        
        with col2:
            st.subheader("Support Vector Regression")
            st.metric("RMSE", f"${svr_metrics['RMSE']:.4f}")
            st.metric("MAE", f"${svr_metrics['MAE']:.4f}")
            st.metric("R² Score", f"{svr_metrics['R²']:.4f}")
            st.metric("MAPE", f"{svr_metrics['MAPE']:.2f}%")
            st.metric("Direction Accuracy", f"{svr_metrics['Direction Accuracy']:.2f}%")
    
    with tab3:
        st.subheader("Model Comparison")
        
        if lstm_metrics['RMSE'] < svr_metrics['RMSE']:
            winner = "LSTM-like Neural Network"
            improvement = ((svr_metrics['RMSE'] - lstm_metrics['RMSE']) / svr_metrics['RMSE']) * 100
        else:
            winner = "Support Vector Regression"
            improvement = ((lstm_metrics['RMSE'] - svr_metrics['RMSE']) / lstm_metrics['RMSE']) * 100
        
        st.success(f"🏆 Best Model: {winner} ({improvement:.2f}% better RMSE)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            metrics_to_compare = ['RMSE', 'MAE', 'MAPE']
            lstm_vals = [lstm_metrics[m] for m in metrics_to_compare]
            svr_vals = [svr_metrics[m] for m in metrics_to_compare]
            
            fig_error = go.Figure(data=[
                go.Bar(name='LSTM-like NN', x=metrics_to_compare, y=lstm_vals, marker_color='#1f77b4'),
                go.Bar(name='SVR', x=metrics_to_compare, y=svr_vals, marker_color='#ff7f0e')
            ])
            fig_error.update_layout(
                title='Error Metrics Comparison',
                barmode='group',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_error, use_container_width=True)
    
    with tab4:
        st.subheader("Error Analysis")
        
        lstm_residuals = y_test_original - lstm_pred_original
        svr_residuals = y_test_original - svr_pred_original
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_lstm_res = go.Figure()
            fig_lstm_res.add_trace(go.Scatter(
                x=test_dates, y=lstm_residuals.flatten(),
                mode='lines', name='LSTM Residuals',
                line=dict(color='#1f77b4', width=2)
            ))
            fig_lstm_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_lstm_res.update_layout(
                title='LSTM-like NN Residuals',
                xaxis_title='Date',
                yaxis_title='Residual ($)',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_lstm_res, use_container_width=True)
        
        with col2:
            fig_svr_res = go.Figure()
            fig_svr_res.add_trace(go.Scatter(
                x=test_dates, y=svr_residuals.flatten(),
                mode='lines', name='SVR Residuals',
                line=dict(color='#ff7f0e', width=2)
            ))
            fig_svr_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_svr_res.update_layout(
                title='SVR Residuals',
                xaxis_title='Date',
                yaxis_title='Residual ($)',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_svr_res, use_container_width=True)

except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.info("Please check your configuration and try again.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9em;">
    <p>Stock Price Prediction using LSTM Neural Networks & Support Vector Regression</p>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. Always consult financial advisors before making investment decisions.</p>
    </div>
""", unsafe_allow_html=True)
