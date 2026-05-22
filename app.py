import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page layout to wide
st.set_page_config(page_title="Weather Data Analysis & Prediction", layout="wide")

# ==========================================
# 1. SYNTHETIC DATA GENERATOR (For Instant Use)
# ==========================================
def generate_synthetic_data():
    """Generates a synthetic weather dataset with realistic seasonal trends."""
    np.random.seed(42)
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2026, 1, 1)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate daily temperature using a sine wave to simulate seasonality (peaks in July, dips in January)
    day_of_year = dates.dayofyear
    base_temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 130) / 365)
    # Add noise and a slight annual warming trend (0.2 degrees per year)
    years_passed = (dates.year - 2021)
    noise = np.random.normal(0, 3, len(dates))
    temperature = base_temp + (years_passed * 0.2) + noise
    
    # Humidity (generally inversely proportional to temperature with noise)
    humidity = np.clip(80 - (temperature - 15) * 1.5 + np.random.normal(0, 8, len(dates)), 20, 100)
    
    # Wind Speed
    wind_speed = np.clip(10 + np.random.normal(0, 4, len(dates)), 1, 40)
    
    df = pd.DataFrame({
        'Date': dates,
        'Temperature': np.round(temperature, 1),
        'Humidity': np.round(humidity, 1),
        'Wind_Speed': np.round(wind_speed, 1)
    })
    return df

# Initialize data
if 'weather_df' not in st.session_state:
    st.session_state['weather_df'] = generate_synthetic_data()

# ==========================================
# 2. APP HEADER & SIDEBAR
# ==========================================
st.title("⛅ Weather Data Analysis & Prediction App")
st.write(
    "Analyze historical weather trends and build simple regression models "
    "to estimate temperature patterns based on time-series inputs."
)

st.sidebar.header("📁 Data Source")
upload_option = st.sidebar.radio("Select Dataset:", ["Use Built-in Weather Data", "Upload Custom CSV"])

if upload_option == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file (must contain 'Date' and 'Temperature' columns)", type=["csv"])
    if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)

        # Flexible date parsing
        df_uploaded['Date'] = pd.to_datetime(
            df_uploaded['Date'],
            format='mixed',
            dayfirst=False,
            errors='coerce'
        )

        # Remove invalid dates
        df_uploaded = df_uploaded.dropna(subset=['Date'])

        if df_uploaded.empty:
                st.sidebar.error("No valid dates found in uploaded CSV.")
                st.stop()

            # Convert temperature safely
            df_uploaded['Temperature'] = pd.to_numeric(
                df_uploaded['Temperature'],
                errors='coerce'
            )

            # Remove invalid temperature rows
            df_uploaded = df_uploaded.dropna(subset=['Temperature'])

            # Stop if all temperatures are invalid
            if df_uploaded.empty:
                st.sidebar.error("No valid temperature values found in uploaded CSV.")
                st.stop()

        # Ensure proper columns
        required_cols = {'Date', 'Temperature'}

        if required_cols.issubset(df_uploaded.columns):
            st.session_state['weather_df'] = df_uploaded
            st.sidebar.success("Custom data loaded successfully!")
        else:
            st.sidebar.error(
                "Error: CSV must contain 'Date' and 'Temperature' columns."
            )

    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

df = st.session_state['weather_df'].copy()

# Ensure the Date column is sorted
df = df.sort_values('Date').reset_index(drop=True)

# Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfYear'] = df['Date'].dt.dayofyear

# ==========================================
# 3. INTERACTIVE DASHBOARD tabs
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "⚙️ Model Training", "🔮 Temperature Prediction"])

# ----------------- TAB 1: EDA -----------------
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Avg Temp (°C)", f"{df['Temperature'].mean():.1f}")
    if 'Humidity' in df.columns:
        col3.metric("Avg Humidity (%)", f"{df['Humidity'].mean():.1f}")
    if 'Wind_Speed' in df.columns:
        col4.metric("Avg Wind Speed (km/h)", f"{df['Wind_Speed'].mean():.1f}")
        
    # Date Range Selector
    st.subheader("Historical Weather Trends")
    min_date = df['Date'].min().to_pydatetime()
    max_date = df['Date'].max().to_pydatetime()
    
    date_range = st.slider("Select Date Range to Visualize:", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    filtered_df = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]
    
    # Target Selection for Plotting
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['Year', 'Month', 'Day', 'DayOfYear']]
    plot_target = st.selectbox("Select metric to plot:", numeric_cols)
    
    fig = px.line(filtered_df, x='Date', y=plot_target, title=f"{plot_target} Over Time", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal Analysis
    st.subheader("Monthly Averages (Seasonality)")
    monthly_avg = df.groupby('Month')[numeric_cols].mean().reset_index()
    fig_bar = px.bar(monthly_avg, x='Month', y='Temperature', labels={'Month': 'Month of Year', 'Temperature': 'Average Temp (°C)'},
                     title="Average Temperature by Month", color='Temperature', color_continuous_scale='plasma')
    fig_bar.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), 
                                      ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))
    st.plotly_chart(fig_bar, use_container_width=True)

# ----------------- TAB 2: Model Training -----------------
with tab2:
    st.header("Model Training & Performance Evaluation")
    st.write("We use features derived from the date (`Year`, `Month`, `Day`, `DayOfYear`) to train a regression model to estimate temperatures.")
    
    # Model configuration
    st.subheader("Model Configuration")
    col_cfg1, col_cfg2 = st.columns(2)
    
    with col_cfg1:
        model_type = st.selectbox("Choose Regression Model:", ["Linear Regression", "Random Forest Regressor"])
    with col_cfg2:
        test_size = st.slider("Test Set Split Ratio (%)", min_value=10, max_value=50, value=20, step=5) / 100
        
    # Model preparation
    # Features (X) and Target (y)
    features = ['Year', 'Month', 'Day', 'DayOfYear']
    X = df[features]
    y = df['Temperature']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False) # Keep time sequence intact
    
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    st.subheader("Model Metrics (On Testing Partition)")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} °C")
    col_m2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    col_m3.metric("R-Squared (R²)", f"{r2:.2f}")
    
    # Visualizing predictions vs true values
    st.subheader("Visualizing Fit")
    test_dates = df.loc[y_test.index, 'Date']
    comparison_df = pd.DataFrame({
        'Date': test_dates,
        'Actual Temperature': y_test,
        'Predicted Temperature': y_pred_test
    }).sort_values('Date')
    
    fig_comp = px.line(comparison_df, x='Date', y=['Actual Temperature', 'Predicted Temperature'], 
                       title="Actual vs. Predicted Temperature (Test Data)", template="plotly_white")
    st.plotly_chart(fig_comp, use_container_width=True)

# ----------------- TAB 3: Prediction -----------------
with tab3:
    st.header("Make Predictions")
    st.write("Pick a date below to let the trained model predict the expected temperature.")
    
    col_pred1, col_pred2 = st.columns(2)
    
    with col_pred1:
        prediction_date = st.date_input("Target Date for Prediction", value=datetime.today() + timedelta(days=1))
        
    with col_pred2:
        pred_year = prediction_date.year
        pred_month = prediction_date.month
        pred_day = prediction_date.day
        # Calculate Day of Year manually
        pred_doy = prediction_date.timetuple().tm_yday
        
        # Prepare feature vector
        input_data = pd.DataFrame([[pred_year, pred_month, pred_day, pred_doy]], columns=['Year', 'Month', 'Day', 'DayOfYear'])
        
        # Predict
        predicted_temp = model.predict(input_data)[0]
        
        st.markdown(f"### Predicted Temperature:")
        st.markdown(f"## <span style='color:#FF4B4B'>{predicted_temp:.2f} °C</span>", unsafe_allow_html=True)
        st.info(f"Using trained model: **{model_type}**")
