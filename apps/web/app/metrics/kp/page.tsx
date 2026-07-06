'use client';

import { useEffect, useState, useCallback } from "react";

import Metrics from "../_components/Metrics";

export default function Forecast() {
  const [metricsData, setMetricsData] = useState<Array<any> | []>([]);
  const [loading, setLoading] = useState(true);

  const loadHmfForecastMetrics = useCallback(async () => {
    try {
      const response = await fetch((process.env.NEXT_PUBLIC_API_POINT || "") + "/api/v1/public/forecast/kp/metrics");

      if (!response.ok) {
        throw new Error("Failed to fetch forecast");
      }

      const metricsResult = await response.json();
      setMetricsData(metricsResult.data);
    } catch (err) {
      console.log(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHmfForecastMetrics();
  }, [loadHmfForecastMetrics]);

  return (
    <div className="container">
      <h1 className="heading">Kp Forecast Metrics</h1>
      <Metrics data={metricsData} labels={{ "kp_threshold_4": "Kp 4", "kp_threshold_5": "Kp 5", "kp_threshold_6": "Kp 6" }}/>
    </div>
  );
}
