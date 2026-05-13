'use client';

import { useEffect, useState, useCallback } from "react";

import WindChart from "./_components/WindChart";
import { prepareChartData } from "./_utils/transform";
import ProbabilityPanel from "./_components/ProbabilityPanel";
import RiskPanel from "./_components/RiskPanel";
// import { forecast } from "./data/mockData";
import styles from "./forecast.module.css"

export default function Forecast() {
  const [forecast, setForecast] = useState<[]>([]);
  const [chartData, setChartData] = useState<[]>([]);
  const [loading, setLoading] = useState(true);

  const loadForecast = useCallback(async () => {
    try {
      setLoading(true);

      const response = await fetch("http://127.0.0.1:8000/api/forecast");

      if (!response.ok) {
        throw new Error("Failed to fetch forecast");
      }

      const forecastResult = await response.json();
      setForecast(forecastResult.forecast);

      const chartData = prepareChartData(forecastResult.forecast);
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
    <div className={styles.forecast}>
      <h1>Argus Sunwatch</h1>

      <RiskPanel data={forecast} />

      <ProbabilityPanel data={forecast} />
      
      <WindChart data={chartData} />
    </div>
  );
}
