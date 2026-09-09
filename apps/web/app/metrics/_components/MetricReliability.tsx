import {
    ResponsiveContainer,
    LineChart,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
    Legend,
    Label,
    Line
} from "recharts";

import { useId, useState } from "react";

import "../../_components/charts.css"
import ContentBlock from "../../_components/ContentBlock";

type ReliabilityRow = {
  x: number;
  values: Record<string, ReliabilityPoint[]>;
};

type ReliabilityPoint = { predicted_probability: number; observed_frequency: number };

type Labels = Record<string, string>;

const linesMeta: Array<any> = [
  {
    "color": "#56B4E9",
    "stroke": "#56B4E9",
    "fontWeight": "400"
  },
  {
    "color": "#009E73",
    "stroke": "#009E73",
    "fontWeight": "600"
  },
  {
    "color": "#D55E00",
    "stroke": "#D55E00",
    "fontWeight": "400"
  },
];

function parseReliability(reliability: ReliabilityPoint[]) {
  return reliability.map((point) => ({
    predicted: point.predicted_probability,
    observed: point.observed_frequency,
  }));
}

function buildReliabilityChartData(
  rows: ReliabilityRow[],
  hour: number,
  labels: Labels
) {
  const row = rows.find((item) => item.x === hour);

  if (!row) return [];

  const selectedKeys = Object.keys(labels);

  const allPoints = selectedKeys.flatMap((key) =>
    parseReliability(row.values[key] ?? []).map((point) => ({
      ...point,
      key,
    }))
  );

  const predictedBins = [...new Set(allPoints.map((p) => p.predicted))].sort(
    (a, b) => a - b
  );

  return predictedBins.map((predicted) => {
    const chartRow: Record<string, number> = {
      predicted,
      perfect: predicted,
    };

    for (const key of selectedKeys) {
      const point = parseReliability(row.values[key] ?? []).find(
        (p) => p.predicted === predicted
      );

      chartRow[key] = point?.observed ?? NaN;
    }

    return chartRow;
  });
}

export default function ReliabilityChart({ data, title, labels }: {data: Array<any>, title: string, labels: Labels})
{
    const [hour, setHour] = useState(1);
    const sliderId = useId();

    if (!data || data.length == 0) {
        return (
        <div>
            <h3 className="heading">{ title }</h3>
            <p>No metrics available.</p>
        </div>
        );
    }

    const leadHours = data.length;

    const xAxisMeta = Array.from({length: leadHours}, (_, i) => i + 1);

    const rechartsData = xAxisMeta.map((x, i) => ({
        x,
        values: data[i],
    }));

    const chartData = buildReliabilityChartData(rechartsData, hour, labels);

    return (
      <ContentBlock>
        <div className="reliability-container">
            <h3 className="heading">{ title }</h3>

            <div style={{ marginBottom: 16 }}>
                <label htmlFor={sliderId} className="color-default">Lead hour: <strong>{hour}</strong></label>

                <input
                    id={sliderId}
                    type="range"
                    min={1}
                    max={leadHours}
                    step={1}
                    value={hour}
                    onChange={(event) => setHour(Number(event.target.value))}
                    style={{ width: "100%" }}
                />
            </div>

            <ResponsiveContainer width="100%" aspect={1}>
                <LineChart data={chartData} margin={{ top: 12, right: 12, bottom: 24, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" />

                    <XAxis
                      tick={{ fill: "#b9aec7", fontSize: 12 }}
                      dataKey="predicted"
                      type="number"
                      domain={[0, 1]}
                      tickFormatter={(v) => `${v}`}
                    >
                      <Label
                        style={{
                            textAnchor: "middle",
                            fontSize: 12,
                            fill: "white",
                        }}
                      angle={0} 
                      position="insideBottom" offset={-12}
                      value={"Predicted probability"} />
                    </XAxis>

                    <YAxis
                      tick={{ fill: "#b9aec7", fontSize: 12 }}
                      type="number"
                      domain={[0, 1]}
                    >
                      <Label
                        style={{
                            textAnchor: "middle",
                            fontSize: 12,
                            fill: "white",
                        }}
                      angle={270} 
                      position="insideLeft"
                      value={"Observed probability"} />
                    </YAxis>

                    {/* <Tooltip
                        cursor={true}
                        animationDuration={0}
                        animationEasing="linear"
                        contentStyle={{
                            backgroundColor: '#18181b',
                            border: '1px solid #3f3f46',
                            borderRadius: '8px',
                            padding: '10px 14px',
                            color: '#e4e4e7',
                            fontSize: '13px',
                            boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.3)',
                        }}
                    /> */}



                    <Line
                        dataKey="perfect"
                        name="Perfect calibration"
                        strokeDasharray="5 5"
                        dot={false}
                        isAnimationActive={false}
                    />

                    {Object.keys(labels).map((key, i) => {
                        const meta = linesMeta[i];

                        return (
                            <Line
                                key={key}
                                dataKey={key}
                                name={labels[key]}
                                stroke={meta["stroke"]}
                                strokeWidth={2.5}
                                fill="none"
                                dot
                                activeDot={{ 
                                    r: 6, 
                                    fill: meta["stroke"],
                                    stroke: '#fff', 
                                    strokeWidth: 2 
                                }}
                                connectNulls
                                isAnimationActive={false}
                            />
                        );
                    })}
                </LineChart>
            </ResponsiveContainer>
            <div className="reliability-legend">
              <span style={{ color: "#8884d8" }}>– – Perfect calibration</span>
              {Object.entries(labels).map(([key, label], index) => <span key={key} style={{ color: linesMeta[index].color }}>● {label}</span>)}
            </div>
        </div>
      </ContentBlock>
    );
}
