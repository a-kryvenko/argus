"use client";

import { useEffect, useState } from "react";

import type { ProductConfig } from "../../_config/products";
import { productApiPath } from "../../_config/products";
import { apiRequest, type ForecastMetrics } from "../../_utils/api";
import ContinuousMetrics from "./ContinuousMetrics";
import Metrics from "./Metrics";

export default function MetricsProduct({ product }: { product: ProductConfig }) {
  const [metrics, setMetrics] = useState<ForecastMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiRequest<ForecastMetrics>(productApiPath(product, "/metrics"))
      .then(setMetrics)
      .catch(err => setError(err instanceof Error ? err.message : "Failed to load metrics"));
  }, [product]);

  return (
    <main className="container color-default">
      <h1 className="heading">{product.title} Metrics</h1>
      <p className="product-description">Metrics by forecast lead hour.</p>
      <nav className="api-links" aria-label="Product metrics API">
        <a href={`/api/v1${productApiPath(product, '/metrics')}`}>Metrics API (JSON)</a>
      </nav>

      {error && <div className="state-message">{error}</div>}
      {!metrics && !error && <div className="state-message">Loading metrics…</div>}

      {metrics && product.variables.map(variable => {
        const variableMetrics = metrics.variables[variable.key];
        if (!variableMetrics) {
          return <div className="state-message" key={variable.key}>{variable.label} metrics are not ready.</div>;
        }
        const labels = Object.fromEntries(variable.thresholds.map(item => [String(item.value), item.label]));
        return (
          <section key={variable.key}>
            <h2 className="variable-heading">{variable.label}</h2>
            <ContinuousMetrics data={metrics} variable={variable.key} label={variable.label} />
            {variableMetrics.binary.length > 0 && (
              <Metrics data={metrics} variable={variable.key} labels={labels} />
            )}
          </section>
        );
      })}
    </main>
  );
}
