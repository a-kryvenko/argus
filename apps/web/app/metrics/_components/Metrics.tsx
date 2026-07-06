import MetricChart from "../_components/MetricChart";

import transformMetrics from "../_utils/transform"
import ReliabilityChart from "./MetricReliability";

export default function Metrics({ data, labels }: {data: Record<string, any>, labels: Record<string, string>})
{   
    return (
        <div>
            <MetricChart data={transformMetrics(data, "brier")} title={ "Brier Score" } labels={labels}/>
            <MetricChart data={transformMetrics(data, "roc_auc")} title={ "ROC AUC" } labels={labels}/>
            <MetricChart data={transformMetrics(data, "avg_precision")} title={ "Precision" } labels={labels}/>
            <ReliabilityChart data={transformMetrics(data, "reliability")} title={ "Reliability" } labels={labels} />
        </div>
    )
}