'use client';

import { useEffect, useState, useCallback } from "react";

import Metrics from "../_components/Metrics";

export default function Forecast() {
  const [metricsData, setMetricsData] = useState<Array<any> | []>([]);
  const [loading, setLoading] = useState(true);

  const loadHmfForecastMetrics = useCallback(async () => {
    try {
      const response = await fetch((process.env.NEXT_PUBLIC_API_POINT || "") + "/api/v1/public/forecast/solar-wind/metrics");

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
      <h1 className="heading">Solar Wind Speed Forecast Metrics</h1>

      <Metrics data={metricsData} labels={{ "v_450": "V > 450 km/s", "v_500": "V > 500 km/s", "v_600": "V > 600 km/s" }}/>
    </div>
  );
}
