# VMarket - Technology Stack & Tools Used

## 🐍 **Core Programming**
- **Python 3.13** - Main programming language

---

## 📊 **Machine Learning & Statistics**

### **Statistical Models**
- **ARCH** - GARCH(1,1) volatility modeling
- **NumPy** - Numerical computations
- **SciPy** - Scientific computing

### **Deep Learning**
- **PyTorch** - LSTM neural network framework
- **CUDA** - GPU acceleration (RTX 2050)
- **torch.nn** - Neural network layers
- **torch.optim** - Model optimization

### **NLP & Sentiment Analysis**
- **Transformers** (Hugging Face) - FinBERT model
- **FinBERT** - Financial sentiment analysis
- **Tokenizers** - Text preprocessing

---

## 📈 **Data Processing**

### **Data Libraries**
- **Pandas** - Data manipulation & analysis
- **NumPy** - Array operations
- **yfinance** - Yahoo Finance data fetching

### **Financial Calculations**
- **Garman-Klass volatility** - Advanced volatility measure
- **Log returns** - Price change calculations
- **Feature engineering** - Custom indicators

---

## 🎨 **Visualization & Dashboard**

### **Dashboard Framework**
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive charts
  - Candlestick charts
  - Line charts
  - Heatmaps
  - Scatter plots

### **UI Components**
- **Streamlit widgets** - Selectbox, date input, tabs
- **Custom CSS** - Dark theme styling
- **Responsive layout** - Multi-column design

---

## 💾 **Data Storage**

### **File Formats**
- **CSV** - Stock data, predictions, results
- **PyTorch (.pth)** - Trained LSTM models
- **Pickle (.pkl)** - GARCH model persistence

### **Data Management**
- **File I/O** - Reading/writing data
- **Index management** - DateTime indexing

---

## 🔧 **Development Tools**

### **Package Management**
- **pip** - Python package installer
- **requirements.txt** - Dependency management

### **Version Control**
- **.gitignore** - File exclusion
- **Git** - Version control system

### **Environment**
- **PowerShell** - Windows terminal
- **.env** - Environment variables (API keys)

---

## 🌐 **APIs & Data Sources**

### **Financial Data**
- **Yahoo Finance API** (via yfinance)
  - Historical OHLCV data
  - Real-time quotes
  - VIX index data

### **News & Sentiment**
- **News API** - Financial news articles
- **FinBERT** - Pre-trained sentiment model

---

## 🧮 **Mathematical Concepts**

### **Volatility Measures**
- **Garman-Klass** - OHLC-based volatility
- **Parkinson** - High-low volatility
- **Close-to-close** - Simple volatility
- **Annualization** - √252 scaling

### **Statistical Methods**
- **GARCH(1,1)** - Conditional heteroskedasticity
- **Maximum Likelihood** - Parameter estimation
- **Confidence Intervals** - Prediction bounds

### **Machine Learning**
- **LSTM** - Long Short-Term Memory networks
- **Backpropagation** - Gradient descent
- **Adam optimizer** - Adaptive learning rate
- **MSE loss** - Mean squared error

---

## 📦 **Key Python Packages**

```
Core ML/Stats:
- numpy
- pandas
- scipy
- arch
- torch
- transformers

Data & Finance:
- yfinance
- ta-lib (optional)

Visualization:
- streamlit
- plotly

Utilities:
- python-dotenv
- tqdm (progress bars)
```

---

## 🏗️ **Architecture**

### **Model Pipeline**
```
Data Fetching (yfinance)
    ↓
Feature Engineering (pandas/numpy)
    ↓
Sentiment Analysis (FinBERT)
    ↓
Model Training (GARCH + LSTM)
    ↓
Ensemble Prediction (40/60 weighting)
    ↓
Visualization (Streamlit + Plotly)
```

### **Directory Structure**
```
VMarket/
├── src/
│   ├── garch_model.py       (ARCH library)
│   ├── lstm_model.py         (PyTorch)
│   ├── volatility_calculator.py
│   ├── sentiment_analyzer.py (Transformers)
│   └── news_fetcher.py       (requests)
├── models/                   (Trained .pth files)
├── data/                     (CSV files)
├── app.py                    (Streamlit)
└── backtest_models.py
```

---

## 🎯 **Novel Combinations**

What makes it unique:
- **GARCH + LSTM** (classical + modern)
- **Financial ML + NLP** (sentiment integration)
- **GPU on consumer hardware** (accessibility)
- **Real-time dashboard** (production-ready)
- **Open-source** (democratization)

---

## 🔑 **Technical Highlights**

1. **GPU Acceleration:** CUDA-enabled LSTM training
2. **Ensemble Learning:** Multiple model combination
3. **Time Series:** Sequence modeling with LSTM
4. **Volatility Clustering:** GARCH captures this
5. **Sentiment Integration:** FinBERT for market psychology
6. **Interactive UI:** Streamlit + Plotly
7. **Production Ready:** Model persistence, backtesting

---

**Total Technologies: 20+ libraries, frameworks, and tools**
