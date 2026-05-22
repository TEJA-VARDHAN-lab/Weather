# 🌦️ Weather Data Analysis & Prediction App

An interactive Machine Learning powered weather analytics dashboard built using Streamlit, Plotly, and Scikit-learn.

This application allows users to analyze historical weather trends, train regression models, evaluate prediction performance, and forecast future temperatures using time-series based weather data.

---

# 🚀 Live Demo

🌐 Project Link:  
http://10.248.224.235:8501

---

# 📌 Features

- 📊 Interactive Weather Data Visualization
- 📁 Upload Custom Weather CSV Files
- 🌡️ Temperature Trend Analysis
- 💧 Humidity & Wind Speed Insights
- 📅 Seasonal & Monthly Weather Analysis
- 🤖 Machine Learning Model Training
- 📈 Performance Metrics Visualization
- 🔮 Future Temperature Prediction
- ⚡ Real-Time Interactive Dashboard

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core Programming Language |
| Streamlit | Web Application Framework |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Plotly | Interactive Charts |
| Scikit-learn | Machine Learning Models |

---

# 📂 Project Structure

```bash
Weather-Data-Analysis/
│
├── app.py                 # Main Streamlit Application
├── requirements.txt       # Required Python Libraries
├── README.md              # Project Documentation
└── sample_weather.csv     # Optional Sample Dataset
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone <your-repository-url>
cd Weather-Data-Analysis
```

---

## 2️⃣ Create Virtual Environment (Optional but Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

# 📊 Application Workflow

## 🔹 1. Synthetic Weather Data Generation

The app automatically generates realistic synthetic weather data including:

- Temperature
- Humidity
- Wind Speed
- Seasonal Trends
- Long-Term Temperature Variations

---

## 🔹 2. Exploratory Data Analysis (EDA)

Users can:

- Visualize historical trends
- Filter data using date sliders
- Compare weather parameters
- Analyze monthly seasonality
- View interactive charts

---

## 🔹 3. Machine Learning Model Training

Supported Regression Models:

- Linear Regression
- Random Forest Regressor

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-Squared Score (R²)

---

## 🔹 4. Temperature Prediction

Users can:

- Select future dates
- Predict expected temperature
- View prediction instantly
- Compare trained model outputs

---

# 📁 CSV Upload Format

Custom datasets must include:

| Column Name | Required |
|---|---|
| Date | ✅ |
| Temperature | ✅ |

### Optional Columns

- Humidity
- Wind_Speed

### Example

```csv
Date,Temperature,Humidity,Wind_Speed
2025-01-01,22.5,65,12
2025-01-02,24.1,60,10
```

---

# 🧠 Machine Learning Pipeline

The application performs:

- Feature Engineering
- Train-Test Split
- Regression Model Training
- Prediction Generation
- Performance Evaluation

### Engineered Features

```python
['Year', 'Month', 'Day', 'DayOfYear']
```

---

# 📷 Dashboard Highlights

## 📊 EDA Dashboard

- Interactive Time-Series Charts
- Seasonal Trend Analysis
- KPI Metrics

## ⚙️ Model Training Dashboard

- Model Selection
- Performance Evaluation
- Actual vs Predicted Visualization

## 🔮 Prediction Dashboard

- Future Temperature Forecasting
- Real-Time Prediction Output

---

# 📦 Requirements

```txt
streamlit
pandas
numpy
plotly
scikit-learn
```

---

# 🔒 Future Improvements

- 🌍 Real Weather API Integration
- 📡 Live Weather Forecasting
- 📱 Mobile Responsive UI
- ☁️ Cloud Deployment
- 🧠 Advanced Deep Learning Models
- 📈 Multi-Variable Forecasting

---

# 👨‍💻 Author

Developed with ❤️ using Python and Streamlit.

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you like this project:

- ⭐ Star the repository
- 🍴 Fork the project
- 🛠️ Contribute improvements
- 📢 Share with others

---
"# Weather" 
