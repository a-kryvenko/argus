'use client';

import { useEffect, useState, useCallback } from "react";

import WindChart from "./components/WindChart";
import { prepareChartData } from "./utils/transform";
import RiskStrip from "./components/RiskStrip"
import ProbabilityPanel from "./components/ProbabilityPanel";
// import { forecast } from "./data/mockData";

function App() {
  const [forecast, setForecast] = useState<[]>([]);
  const [chartData, setChartData] = useState<[]>([]);
  const [loading, setLoading] = useState(false);

  const loadForecast = useCallback(async () => {
    try {
      setLoading(true);

      const response = await fetch("http://127.0.0.1:8000/api/forecast");

      if (!response.ok) {
        throw new Error("Failed to fetch forecast");
      }

      const forecastResult = await response.json();
      setForecast(forecastResult.points);

      const chartData = prepareChartData(forecastResult.points);
      setChartData(chartData);
    } catch (err) {
      console.log(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadForecast();
  }, [loadForecast])

  return (
    <div style={{ padding: 20 }}>
      <h1>Argus Sunwatch</h1>

      {loading && <p>Loading...</p>}

      <WindChart data={chartData} />

      <h2>Risk Overview</h2>

      <RiskStrip
        data={forecast}
        label="KP Risk"
        keyName="kp_risk"
      />

      <RiskStrip
        data={forecast}
        label="Satellite Drag Risk"
        keyName="drag_risk_proxy"
      />

      <ProbabilityPanel data={forecast} />

    </div>
  );
}

export default App;