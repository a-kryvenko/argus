'use client';

import { useEffect, useState, useCallback } from "react";

import Metrics from "../_components/Metrics";

export default function Forecast() {
  const [metricsData, setMetricsData] = useState<Array<any> | []>([]);
  const [loading, setLoading] = useState(true);

  const loadHmfForecastMetrics = useCallback(async () => {
    try {
      const response = await fetch((process.env.NEXT_PUBLIC_API_POINT || "") + "/api/v1/private/forecast/hmf/metrics");

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
      <h1 className="heading">HMF Forecast Metrics</h1>
      <Metrics data={metricsData} labels={{ "bt_threshold_5": "HMF > 5 nTl", "bt_threshold_10": "HMF > 10 nTl", "bt_threshold_15": "HMF > 15 nTl" }}/>
    </div>
  );
}
