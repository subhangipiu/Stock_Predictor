import { useState } from "react";

function StockForm({ onPredict }) {
  const [ticker, setTicker] = useState("AAPL");
  const [epochs, setEpochs] = useState(5);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!ticker) return;
    onPredict(ticker.toUpperCase(), epochs || 5);
  };

  return (
    <>
      <p style={{ color: "#fff", marginBottom: "10px", textAlign: "center" }}>
        <strong>Note:</strong> The default currency is USD. For Indian stocks,
        add <code>.NS</code> at the end of the symbol (e.g.,{" "}
        <code>RELIANCE.NS</code>).
      </p>
      <form onSubmit={handleSubmit} className="stock-form">
        <input
          type="text"
          placeholder="Enter Stock Symbol (e.g. AAPL)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
        />
        <input
          type="number"
          placeholder="Enter number of training cycles (e.g. 5)"
          value={epochs}
          onChange={(e) => setEpochs(e.target.value)}
        />
        <button type="submit">Predict</button>
      </form>
    </>
  );
}

export default StockForm;
