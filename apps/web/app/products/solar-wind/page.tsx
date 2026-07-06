'use client';

import { useEffect, useState, useCallback } from "react";

import WindChart from "../../_components/WindChart";
import { prepareWindChartData, preparePlasmaHeatmapData, prepareKpHeatmapData } from "../../_utils/transform";
import HeatMap from "../../_components/HeatMap"
import styles from "../../page.module.css"

export default function Forecast() {
  const [windChartData, setWindChartData] = useState<Array<any> | []>([]);
  const [windProbabilityData, setWindProbabilityData] = useState<Array<any> | []>([]);
  const [loading, setLoading] = useState(true);

  const loadPlasmaForecast = useCallback(async () => {
    try {
      const response = await fetch((process.env.NEXT_PUBLIC_API_POINT || "") + "/api/v1/public/forecast/solar-wind");

      if (!response.ok) {
        throw new Error("Failed to fetch forecast");
      }

      const forecastResult = await response.json();
      setWindChartData(prepareWindChartData(forecastResult.data.points));
      setWindProbabilityData(preparePlasmaHeatmapData(forecastResult.data.points));
    } catch (err) {
      console.log(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPlasmaForecast();
  }, [loadPlasmaForecast])

  return (
    <div className="container">
      <div className={styles.forecast}>
        <HeatMap title="Solar Wind Probability" yLabels={["450 km/s", "500 km/s", "600 km/s"]} data={windProbabilityData} />

        <WindChart data={windChartData} />
      </div>
    </div>
  );
}
