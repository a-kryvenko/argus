'use client';

import { useEffect, useState, useCallback } from "react";

import WindChart from "./_components/WindChart";
import { prepareWindChartData, preparePlasmaHeatmapData, prepareKpHeatmapData } from "./_utils/transform";
import HeatMap from "./_components/HeatMap"
import { apiRequest, type Forecast as ForecastData } from "./_utils/api";

export default function Forecast() {
  const [windChartData, setWindChartData] = useState<Array<any> | []>([]);
  const [windProbabilityData, setWindProbabilityData] = useState<Array<any> | []>([]);
  const [kpRiskData, setKpRiskData] = useState<Array<any> | []>([]);
  const [loading, setLoading] = useState(true);

  const loadPlasmaForecast = useCallback(async () => {
    try {
      const forecast = await apiRequest<ForecastData>("/public/forecasts/solar-wind-speed");
      setWindChartData(prepareWindChartData(forecast.predictions));
      setWindProbabilityData(preparePlasmaHeatmapData(forecast.predictions));
    } catch (err) {
      console.log(err);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadKpRiskForecast = useCallback(async () => {
    try {
      const forecast = await apiRequest<ForecastData>("/public/forecasts/geomagnetic-activity");
      setKpRiskData(prepareKpHeatmapData(forecast.predictions));
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
      <h1 className="text-center">Argus Sunwatch</h1>
      <h2 className="text-center">Solar activity forecast & Impact Intelligence</h2>
      <div>
        <HeatMap title="Kp Index" yLabels={["Kp 4", "Kp 5", "Kp 6"]} data={kpRiskData} />

        <HeatMap title="Solar Wind" yLabels={["450 km/s", "500 km/s", "600 km/s"]} data={windProbabilityData} />

        <WindChart data={windChartData} />
      </div>
    </div>
  );
}
