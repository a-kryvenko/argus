import MetricChart from "../_components/MetricChart";

import transformMetrics from "../_utils/transform"
import ReliabilityChart from "./MetricReliability";
import type { ForecastMetrics } from "../../_utils/api";

export default function Metrics({ data, variable, labels }: {data: ForecastMetrics | null, variable: string, labels: Record<string, string>})
{   
    if (!data) return <p>Loading...</p>;

    return (
        <div>
            <MetricChart data={transformMetrics(data, variable, "brier_score")} title={ "Brier Score" } labels={labels}/>
            <MetricChart data={transformMetrics(data, variable, "roc_auc")} title={ "ROC AUC" } labels={labels}/>
            <MetricChart data={transformMetrics(data, variable, "average_precision")} title={ "Precision" } labels={labels}/>
            <ReliabilityChart data={transformMetrics(data, variable, "reliability")} title={ "Reliability" } labels={labels} />
        </div>
    )
}
