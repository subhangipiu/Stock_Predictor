# Stock Predictor

Stock Predictor is a web application that uses AI and machine learning to analyze and predict stock prices.  
It features a Flask backend for model training and prediction, and a modern frontend for user interaction.

## Features
- Fetches historical and live stock data
- Visualizes stock trends with interactive charts
- Trains deep learning models to predict future prices
- Displays prediction accuracy (RMSE) and next-day forecasts
- User-friendly interface for entering stock details

## Technologies Used
- Python (Flask, TensorFlow, scikit-learn, pandas)
- JavaScript (React for frontend)
- Plotly for charting
- Streamlit for interactive dashboards

---

## Installation & Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/subhangipiu/Stock_Predictor.git
cd Stock_Predictor
```

---

### 2. Backend Setup (Flask + Python)

#### a. Install Python

- Download and install Python from [python.org](https://www.python.org/downloads/)
- During installation, **check "Add Python to PATH"**

#### b. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Activate virtual environment (Mac/Linux)
source venv/bin/activate
```

#### c. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

#### d. Run the Backend Server

```bash
python app.py
```
- The backend will start at `http://127.0.0.1:5000/`

---

### 3. Frontend Setup (React)

#### a. Navigate to Frontend Directory

```bash
cd frontend
```

#### b. Install Node.js & npm

- Download and install Node.js (includes npm) from [nodejs.org](https://nodejs.org/)

#### c. Install Frontend Dependencies

```bash
npm install
```

#### d. Run the Frontend Development Server

```bash
npm run dev
```
- The frontend will start at `http://localhost:3000/` (or as shown in your terminal)

---

## How to Use

1. Open the frontend in your browser (`http://localhost:3000/`)
2. Enter a stock name and date range.
3. View historical data and live price.
4. Train the AI model and see predictions.

---

## Troubleshooting

- If you get Python or npm errors, make sure both are installed and added to your system PATH.
- Always activate your Python virtual environment before running backend commands.
- Do not commit `venv/` or `node_modules/` folders (they are ignored by `.gitignore`).

---

*For further help, raise an issue in this repository.*