'use client';

import { useEffect, useState, useCallback } from "react";
import WindSpeedChart from "./_components/WindSpeedChart"
import WindThresholdChart from "./_components/WindThresholdChart";

export default function metricsPage()
{
    const [windMetrics, setWindMetrics] = useState<[]>([]);
    const [thresholdMetrics, setThresholdMetrics] = useState<[]>([]);
    const [loading, setLoading] = useState(true);
    
    const loadMetrics = useCallback(async () => {
        try {
          setLoading(true);
    
          const response = await fetch((process.env.NEXT_PUBLIC_API_POINT || "") + "/api/metrics/all");
    
          if (!response.ok) {
            throw new Error("Failed to fetch metrics");
          }
    
          const metricsResult = await response.json();
          setWindMetrics(metricsResult["wind_speed"]);
          setThresholdMetrics(metricsResult["wind_threshold"]);
        } catch (err) {
          console.log(err);
        } finally {
          setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadMetrics();
    }, [loadMetrics])

    return (
        <div className="container">
        <WindSpeedChart data={ windMetrics }/>
        <WindThresholdChart data={ thresholdMetrics }/>
        </div>
    );
}