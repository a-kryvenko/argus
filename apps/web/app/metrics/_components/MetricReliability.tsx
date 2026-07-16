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

import { useState } from "react";

import "../../_components/charts.css"
import ContentBlock from "../../_components/ContentBlock";

type ReliabilityRow = {
  x: number;
  values: Record<string, string>;
};

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

function parseReliability(reliability: string) {
  return reliability.split(";").map((pair) => {
    const [predicted, observed] = pair.split("_").map(Number);

    return {
      predicted,
      observed,
    };
  });
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
    parseReliability(row.values[key] ?? "").map((point) => ({
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
      const point = parseReliability(row.values[key] ?? "").find(
        (p) => p.predicted === predicted
      );

      chartRow[key] = point?.observed ?? NaN;
    }

    return chartRow;
  });
}

export default function ReliabilityChart({ data, title, labels }: {data: Array<any>, title: string, labels: Labels})
{
    if (!data || data.length == 0) {
        return (
        <div>
            <h3 className="heading">{ title }</h3>
            <p>Loading...</p>
        </div>
        );
    }

    const leadHours = data.length;

    const xAxisMeta = Array.from({length: leadHours}, (_, i) => i + 1);

    const [hour, setHour] = useState(1);

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
                <label className="color-default">Lead hour: <strong>{hour}</strong></label>

                <input
                    type="range"
                    min={1}
                    max={leadHours}
                    step={1}
                    value={hour}
                    onChange={(event) => setHour(Number(event.target.value))}
                    style={{ width: "100%" }}
                />
            </div>

            <ResponsiveContainer width={500} height={500}>
                <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />

                    <XAxis
                      dataKey="predicted"
                      type="number"
                      domain={[0, 1]}
                      tickFormatter={(v) => `${v}`}
                    >
                      <Label
                        style={{
                            textAnchor: "middle",
                            fontSize: "130%",
                            fill: "white",
                        }}
                      angle={0} 
                      value={"Predicted probability"} />
                    </XAxis>

                    <YAxis
                      type="number"
                      domain={[0, 1]}
                    >
                      <Label
                        style={{
                            textAnchor: "middle",
                            fontSize: "130%",
                            fill: "white",
                        }}
                      angle={270} 
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

                    <Legend />

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
        </div>
      </ContentBlock>
    );
}