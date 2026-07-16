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
      <h1 className="heading">Southward Bz Forecast Metrics</h1>
      
      <Metrics data={metricsData} labels={{ "southward_bz_5": "South Bz > 5 nTl", "southward_bz_10": "South Bz > 10 nTl", "southward_bz_15": "South Bz > 15 nTl" }}/>
    </div>
  );
}
