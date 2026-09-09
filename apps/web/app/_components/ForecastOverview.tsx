"use client";
import HeatMap from "./HeatMap";
import WindChart from "./WindChart";
import ResourceState from "./ResourceState";
import { useResource } from "../_utils/useResource";
import type { Forecast } from "../_utils/api";
import {
  formatForecastTime,
  probabilityData,
  quantileData,
} from "../_utils/forecast";

function ForecastSection({
  target,
  variable,
  title,
  thresholds,
  labels,
  wind = false,
}: {
  target: string;
  variable: string;
  title: string;
  thresholds: number[];
  labels: string[];
  wind?: boolean;
}) {
  const { data, error, retry } = useResource<Forecast>(
    `/public/forecasts/${target}`,
  );
  return (
    <section className="forecast-section" aria-label={title}>
      {!data ? (
        <ResourceState
          error={error}
          retry={retry}
          label={title.toLowerCase()}
        />
      ) : (
        <>
          <p className="forecast-meta">
            {title} · Issued{" "}
            <time dateTime={data.issue_time}>
              {formatForecastTime(data.issue_time)}
            </time>
            {data.predictions.length > 0 && (
              <>
                <br />
                Valid {formatForecastTime(data.predictions[0].valid_time)}
                {" – "}
                {formatForecastTime(data.predictions[data.predictions.length - 1].valid_time)}
              </>
            )}
          </p>
          <HeatMap
            title={`${title} threshold probability`}
            yLabels={labels}
            data={probabilityData(data, variable, thresholds)}
            times={data.predictions.map((point) => point.valid_time)}
          />
          {wind && <WindChart data={quantileData(data, variable)} />}
        </>
      )}
    </section>
  );
}
export default function ForecastOverview() {
  return (
    <>
      <ForecastSection
        target="geomagnetic-activity"
        variable="kp"
        title="Kp Index"
        thresholds={[4, 5, 6]}
        labels={["Kp ≥ 4", "Kp ≥ 5", "Kp ≥ 6"]}
      />
      <ForecastSection
        target="solar-wind-speed"
        variable="v"
        title="Solar Wind"
        thresholds={[450, 500, 600]}
        labels={["≥ 450 km/s", "≥ 500 km/s", "≥ 600 km/s"]}
        wind
      />
    </>
  );
}
