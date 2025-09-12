import { useState } from "react";
import StockForm from "./components/StockForm";
import Result from "./components/Result";
import "./index.css";

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async (ticker, epochs) => {
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:5000/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, epochs }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ status: "error", error: error.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="card">
        <h1>📈 Stock Price Predictor</h1>

        <StockForm onPredict={handlePredict} />

        {loading && <p className="loading">⏳ Training model, please wait...</p>}

        {result && <Result data={result} />}
      </div>
    </div>
  );
}

export default App;
