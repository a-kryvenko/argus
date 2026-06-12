'use client';

import { useEffect, useState, useCallback } from "react";

import WindChart from "./_components/WindChart";
import { prepareWindChartData, prepareProbabilitiesData } from "./_utils/transform";
import ProbabilityPanel from "./_components/ProbabilityPanel";
import RiskPanel from "./_components/RiskPanel";
// import { forecast } from "./data/mockData";
import styles from "./page.module.css"

export default function Forecast() {
  const [windChartData, setWindChartData] = useState<Array<any> | []>([]);
  const [windProbabilityData, setWindProbabilityData] = useState<Array<any> | []>([]);
  const [loading, setLoading] = useState(true);

  const loadForecast = useCallback(async () => {
    try {
      const response = await fetch((process.env.NEXT_PUBLIC_API_POINT || "") + "/api/v1/public/forecast/solar-wind");

      if (!response.ok) {
        throw new Error("Failed to fetch forecast");
      }

      const forecastResult = await response.json();
      setWindChartData(prepareWindChartData(forecastResult.data.points));
      setWindProbabilityData(prepareProbabilitiesData(forecastResult.data.points));
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

          <ProbabilityPanel data={windProbabilityData} />

        </div>
        
        <WindChart data={windChartData} />
      </div>
    </div>
  );
}
