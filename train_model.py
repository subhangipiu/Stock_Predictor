# train_model.py
import argparse
import os
import random
import math
import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")        # for server (no GUI)
import matplotlib.pyplot as plt

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------- Reproducibility ----------------
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# ---------------- Helpers ----------------
def create_multivariate_dataset(data, time_step=60):
    x, y = [], []
    for i in range(time_step, len(data)):
        x.append(data[i-time_step:i, :])
        y.append(data[i, 0])  # predict Close
    return np.array(x), np.array(y)

# ---------------- Main Training Function ----------------
def train_and_save(ticker="AAPL", epochs=5, start_date="2020-01-01"):
    ticker = str(ticker).upper()
    end_date = pd.to_datetime("today").strftime("%Y-%m-%d")
    print(f"📥 Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        raise ValueError(f"No data returned for {ticker}")
    if "Close" not in data.columns or "Volume" not in data.columns:
        raise ValueError(f"Missing required columns for {ticker}: {data.columns.tolist()}")

    # ---------------- Currency ----------------
    try:
        ticker_info = yf.Ticker(ticker).info
        currency = ticker_info.get("currency", "USD")
    except Exception:
        currency = "USD"

    features = data[["Close", "Volume"]]
    dataset = features.values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # ---------------- Train/Test Split ----------------
    time_step = 60
    train_size = math.ceil(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size, :]
    test_data = scaled_data[train_size - time_step:, :]

    x_train, y_train = create_multivariate_dataset(train_data, time_step)
    x_test, y_test = create_multivariate_dataset(test_data, time_step)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 2)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 2)

    # ---------------- LSTM Model ----------------
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 2)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    print("🧑‍🏫 Training model...")
    model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1)
    print("✅ Training complete")

    # ---------------- Predictions ----------------
    preds_scaled = model.predict(x_test)
    dummy = np.zeros((len(preds_scaled), 2))
    dummy[:,0] = preds_scaled.flatten()
    preds = scaler.inverse_transform(dummy)[:,0]

    y_test_actual = data["Close"].iloc[train_size:].values
    minlen = min(len(y_test_actual), len(preds))
    y_test_actual = y_test_actual[:minlen]
    preds = preds[:minlen]

    rmse = math.sqrt(mean_squared_error(y_test_actual, preds))
    print(f"\nNew Root Mean Squared Error (RMSE): {rmse:.2f} {currency}")

    # ---------------- Next-day Prediction ----------------
    last_60 = features[-time_step:].values
    last_60_scaled = scaler.transform(last_60)
    X_next = np.array([last_60_scaled])
    pred_next_scaled = model.predict(X_next)
    dummy_pred = np.zeros((1,2))
    dummy_pred[:,0] = pred_next_scaled.flatten()
    next_day_price = scaler.inverse_transform(dummy_pred)[:,0][0]

    print(f"\n🔮 PREDICTION FOR THE NEXT TRADING DAY: {next_day_price:.2f} {currency}")

    # ---------------- Plot Graph ----------------
    train_series = data["Close"].iloc[:train_size]
    valid_series = data["Close"].iloc[train_size:]
    valid_df = pd.DataFrame({"Actual": valid_series.values.flatten()})
    valid_df["Predicted"] = preds

    plt.style.use("dark_background")
    plt.figure(figsize=(12,5))
    plt.plot(train_series.index, train_series.values, label="Training History", color="#636efa")
    plt.plot(valid_series.index, valid_series.values, label="Actual Price", color="#ef553b")
    plt.plot(valid_series.index[:len(preds)], preds, label="Predicted Price", color="#00cc96")
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel(f"Price ({currency})")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # ---------------- Save Model ----------------
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{ticker}_lstm.h5")
    model.save(model_path)

    # ---------------- Return Data ----------------
    return {
        "ticker": ticker,
        "rmse": rmse,
        "next_day_prediction": next_day_price,
        "currency": currency,
        "graph": graph_b64,
        "model_path": model_path
    }

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    result = train_and_save(args.ticker.upper(), epochs=args.epochs)
    print(f"Done -> RMSE={result['rmse']:.2f} {result['currency']}, Next-day={result['next_day_prediction']:.2f} {result['currency']}, model={result['model_path']}")
