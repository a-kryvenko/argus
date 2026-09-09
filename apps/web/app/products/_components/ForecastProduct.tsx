"use client";

import { useResource } from "../../_utils/useResource";
import ResourceState from "../../_components/ResourceState";
import { formatForecastTime, quantileData, probabilityData } from "../../_utils/forecast";

import HeatMap from "../../_components/HeatMap";
import WindChart from "../../_components/WindChart";
import type { ProductConfig } from "../../_config/products";
import { productApiPath } from "../../_config/products";
import { type Forecast } from "../../_utils/api";

export default function ForecastProduct({ product }: { product: ProductConfig }) {
  const { data: forecast, error, retry } = useResource<Forecast>(productApiPath(product));

  return (
    <main className="container color-default ">
      <h1 className="heading">{product.title}</h1>
      <p className="product-description">{product.description}</p>
      <nav className="api-links" aria-label="Product API">
        <a href={`/api/v1${productApiPath(product)}`}>Forecast API (JSON)</a>
        <a href={`/api/v1${productApiPath(product, '/metrics')}`}>Metrics API (JSON)</a>
      </nav>

      {!forecast && <ResourceState error={error} retry={retry} label="forecast" />}
      {forecast && <p className="forecast-meta">Issued <time dateTime={forecast.issue_time}>{formatForecastTime(forecast.issue_time)}</time> · All chart times in UTC</p>}

      {forecast && product.variables.map(variable => {
        const available = forecast.available_variables.includes(variable.key);
        if (!available) {
          return <div className="state-message" key={variable.key}>{variable.label} forecast is not ready.</div>;
        }
        return (
          <section key={variable.key} className="forecast-section">
            {variable.quantile && (
              <WindChart
                data={quantileData(forecast, variable.key)}
                title={`${variable.label} Quantile Forecast`}
                unit={variable.unit}
              />
            )}
            {variable.thresholds.length > 0 && (
              <HeatMap
                title={`${variable.label} Threshold Probability`}
                yLabels={variable.thresholds.map(item => item.label)}
                data={probabilityData(forecast, variable.key, variable.thresholds.map(item => item.value))}
                times={forecast.predictions.map(point => point.valid_time)}
              />
            )}
          </section>
        );
      })}
    </main>
  );
}
