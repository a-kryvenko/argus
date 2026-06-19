'use client';

import { useEffect, useState, useCallback } from "react";

import WindChart from "./_components/WindChart";
import { prepareWindChartData, preparePlasmaHeatmapData, prepareKpHeatmapData } from "./_utils/transform";
import HeatMap from "./_components/HeatMap"
import styles from "./page.module.css"

export default function Forecast() {
  const [windChartData, setWindChartData] = useState<Array<any> | []>([]);
  const [windProbabilityData, setWindProbabilityData] = useState<Array<any> | []>([]);
  const [kpRiskData, setKpRiskData] = useState<Array<any> | []>([]);
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

  const loadKpRiskForecast = useCallback(async () => {
    try {
      const response = await fetch((process.env.NEXT_PUBLIC_API_POINT || "") + "/api/v1/public/forecast/kp");

      if (!response.ok) {
        throw new Error("Failed to fetch forecast");
      }

      const forecastResult = await response.json();
      setKpRiskData(prepareKpHeatmapData(forecastResult.data.points));
    } catch (err) {
      console.log(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPlasmaForecast();
  }, [loadPlasmaForecast])

  useEffect(() => {
    loadKpRiskForecast();
  }, [loadKpRiskForecast])

  return (
    <div className="container">
      <div className={styles.forecast}>
        <HeatMap title="Kp Risk Forecast" yLabels={["Kp 4", "Kp 5", "Kp 6"]} data={kpRiskData} />

        <HeatMap title="Solar Wind Probability" yLabels={["450 km/s", "500 km/s", "600 km/s"]} data={windProbabilityData} />

        <WindChart data={windChartData} />
      </div>
    </div>
  );
}
