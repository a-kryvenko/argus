"use client";

import { useEffect, useState } from "react";

import HeatMap from "../../_components/HeatMap";
import WindChart from "../../_components/WindChart";
import type { ProductConfig, ProductVariable } from "../../_config/products";
import { productApiPath } from "../../_config/products";
import { apiRequest, type Forecast } from "../../_utils/api";

function quantileData(forecast: Forecast, variable: ProductVariable) {
  return forecast.predictions.flatMap(point => {
    const continuous = point.variables[variable.key]?.continuous;
    if (!continuous) return [];
    return [{
      time: point.valid_time,
      low: continuous.q10,
      median: continuous.q50,
      high: continuous.q90,
    }];
  });
}

function probabilityData(forecast: Forecast, variable: ProductVariable) {
  const rows: Array<Array<number>> = [];
  forecast.predictions.forEach((point, x) => {
    const values = point.variables[variable.key]?.binary ?? [];
    variable.thresholds.forEach((threshold, y) => {
      const prediction = values.find(item => item.threshold === threshold.value);
      if (prediction) rows.push([x, y, Math.round(prediction.probability * 100)]);
    });
  });
  return rows;
}

export default function ForecastProduct({ product }: { product: ProductConfig }) {
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiRequest<Forecast>(productApiPath(product))
      .then(setForecast)
      .catch(err => setError(err instanceof Error ? err.message : "Failed to load forecast"));
  }, [product]);

  return (
    <main className="container color-default">
      <h1 className="heading">{product.title}</h1>
      <p className="product-description">{product.description}</p>

      {error && <div className="state-message">{error}</div>}
      {!forecast && !error && <div className="state-message">Loading forecast…</div>}

      {forecast && product.variables.map(variable => {
        const available = forecast.available_variables.includes(variable.key);
        if (!available) {
          return <div className="state-message" key={variable.key}>{variable.label} forecast is not ready.</div>;
        }
        return (
          <section key={variable.key}>
            {variable.quantile && (
              <WindChart
                data={quantileData(forecast, variable)}
                title={`${variable.label} Quantile Forecast`}
                unit={variable.unit}
              />
            )}
            {variable.thresholds.length > 0 && (
              <HeatMap
                title={`${variable.label} Threshold Probability`}
                yLabels={variable.thresholds.map(item => item.label)}
                data={probabilityData(forecast, variable)}
                horizon={forecast.horizon_hours}
              />
            )}
          </section>
        );
      })}
    </main>
  );
}
