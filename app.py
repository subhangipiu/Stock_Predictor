# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from train_model import train_and_save

app = Flask(__name__)
CORS(app)  # dev-only: allow all origins

@app.route("/train", methods=["POST"])
def train_route():
    payload = request.get_json() or {}
    ticker = payload.get("ticker", "AAPL").upper()
    epochs = int(payload.get("epochs", 5))
    try:
        result = train_and_save(ticker=ticker, epochs=epochs)  # get dict
        return jsonify({
            "status": "ok",
            "ticker": result["ticker"],
            "rmse": round(result["rmse"], 2),
            "next_day_prediction": round(result["next_day_prediction"], 2),
            "currency": result.get("currency", "USD"),
            "graph": result["graph"],
            "model_path": result.get("model_path", ""),
            "output": f"Training {ticker} for {epochs} epochs completed."
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "output": f"Failed to train {ticker}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
