'use client';

import { useEffect, useState, useCallback } from "react";

import WindChart from "./_components/WindChart";
import { prepareChartData } from "./_utils/transform";
import ProbabilityPanel from "./_components/ProbabilityPanel";
import RiskPanel from "./_components/RiskPanel";
// import { forecast } from "./data/mockData";
import styles from "./page.module.css"

export default function Forecast() {
  const [forecast, setForecast] = useState<[]>([]);
  const [chartData, setChartData] = useState<Array<any> | []>([]);
  const [loading, setLoading] = useState(true);

  const loadForecast = useCallback(async () => {
    try {
      setLoading(true);

      const response = await fetch((process.env.NEXT_PUBLIC_API_POINT || "") + "/api/forecast/all");

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
    <div className="container">
      <div className={styles.forecast}>
        <div style={{padding: "0 0 0 70px"}}>

          {/* <RiskPanel data={forecast} /> */}

          <ProbabilityPanel data={forecast} />

        </div>
        
        <WindChart data={chartData} />
      </div>
    </div>
  );
}
