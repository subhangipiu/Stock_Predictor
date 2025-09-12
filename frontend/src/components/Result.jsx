import { color } from "framer-motion";

function Result({ data }) {
  if (data.status === "error") {
    return (
      <div className="text-red-400 mt-4 text-center font-semibold color-white;">
        ❌ Error: {data.error}
      </div>
    );
  }

  return (
    <div
      className="mt-6 bg-gray-800 p-6 rounded-2xl shadow-md text-center w-full max-w-5xl"
      style={{ color: "white" }}
    >
      <h2 className="text-2xl font-semibold text-green-400 mb-4">
        ✅ Training Complete
      </h2>
      <p className="mb-2">
        <strong>Ticker:</strong> {data.ticker}
      </p>
      <p className="mb-2">
        <strong>RMSE:</strong> {data.rmse} {data.currency || "USD"}
      </p>
      <p className="mb-4">
        <strong>Next Day Prediction:</strong> {data.next_day_prediction}{" "}
        {data.currency || "USD"}
      </p>

      {data.graph && (
  <div className="graph-wrapper mt-4">
    <img
      src={data.graph}
      alt="Stock Prediction Chart"
      className="rounded-lg shadow-lg mx-auto"
      style={{ width: "100%", maxWidth: "100%", height: "auto", maxHeight: "500px", display: "block" }}
    />
  </div>
)}
    </div>
  );
}

export default Result;
