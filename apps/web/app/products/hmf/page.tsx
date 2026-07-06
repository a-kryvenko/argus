'use client';

import { useEffect, useState, useCallback } from "react";

import { prepareTotalHmfHeatmapData } from "./_utils/transform";
import HeatMap from "../../_components/HeatMap"
import styles from "../../page.module.css"

export default function Forecast() {
  const [btRiskData, setBtRiskData] = useState<Array<any> | []>([]);
  const [loading, setLoading] = useState(true);

  const loadHmfRiskForecast = useCallback(async () => {
    try {
      const response = await fetch((process.env.NEXT_PUBLIC_API_POINT || "") + "/api/v1/private/forecast/hmf");

      if (!response.ok) {
        throw new Error("Failed to fetch forecast");
      }

      const forecastResult = await response.json();
      setBtRiskData(prepareTotalHmfHeatmapData(forecastResult.data.points));
    } catch (err) {
      console.log(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHmfRiskForecast();
  }, [loadHmfRiskForecast]);

  return (
    <div className="container">
      <div className={styles.forecast}>
        <HeatMap title="Heliomagnetic Risk Forecast" yLabels={[">= 5 nTl", ">= 10 nTl", ">= 15 nTl"]} data={btRiskData} />
      </div>
    </div>
  );
}
