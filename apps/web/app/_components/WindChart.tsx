import {
  CartesianGrid,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart
} from "recharts";

import "./charts.css";
import { formatForecastTime } from "../_utils/forecast";
import ContentBlock from "./ContentBlock";

const linesMeta: any = {
  "low": {
    "name": "Low",
    "color": "#b3d0fa",
    "stroke": "transparent",
    "fontWeight": "400"
  },
  "median": {
    "name": "Median",
    "color": "#7480ff",
    "stroke": "#7480ff",
    "fontWeight": "600"
  },
  "high": {
    "name": "High",
    "color": "#b3d0fa",
    "stroke": "transparent",
    "fontWeight": "400"
  },
}

const lineKeys = ["median", "low", "high"] as const;

export default function WindChart({ data, title = "Solar Wind Speed", unit = "km/s" }: {data: any[], title?: string, unit?: string}) {
  if (!data || !data.some(row => row.median !== null && row.median !== undefined)) {
    return (
      <ContentBlock>
        <h2>{title}</h2>
        <p className="forecast-meta" role="status">No quantile forecast available.</p>
      </ContentBlock>
    );
  }

  const rechartsData = data.map((values, i) => ({
    index: i,
    dayName: new Date(values.time).toLocaleDateString(undefined, { weekday: "short", timeZone: "UTC" }),
    hour: new Date(values.time).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", timeZone: "UTC", hourCycle: "h23" }),
    values: data[i],
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) {
      return null;
    }

    const row = payload[0].payload;

    return (
      <div className="tooltip">
        <div className="tooltip__title">
          {formatForecastTime(row.values.time)}
        </div>

        {lineKeys.map((key) => (
          <div
            className="tooltip__row"
            key={key}
            style={{
              color: linesMeta[key].color,
              fontWeight: linesMeta[key].fontWeight,
            }}
          >
            <span>{linesMeta[key].name}</span>
            <span>{row.values[key] ?? "—"} {unit}</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <ContentBlock>
      <h2>{title}</h2>
      <p className="forecast-meta">Median · {unit} · UTC. The tooltip includes the 10th and 90th percentiles.</p>

      <div style={{ height: 400 }}>
        <ResponsiveContainer>
          <LineChart
            data={rechartsData}
            margin={{
              top: 0,
              right: 0,
              bottom: 0,
              left: 0,
            }}
          >
            <XAxis
              tick={{ fill: "#b9aec7", fontSize: 12 }}
              minTickGap={30}
              dataKey="index"
              type="number"
              domain={[0, Math.max(0, rechartsData.length - 1)]}
              ticks={rechartsData
                .filter((_, i) => i % 6 === 0)
                .map((d) => d.index)}
              tickFormatter={(index) => {
                const point = rechartsData[index];
                return point ? `${point.dayName} ${point.hour}` : "";
              }}
            />
            
            <YAxis
              type="number"
              domain={[
                dataMin => Math.floor(dataMin / 10) * 10,
                dataMax => Math.ceil(dataMax / 10) * 10,
              ]}
              tickCount={6}
              width={70}
              tick={{ fontSize: 12, fill: "#b9aec7" }}
            />

            <CartesianGrid strokeDasharray="3 3" stroke="#413449" />

            <Tooltip
              cursor={true}
              animationDuration={0}
              animationEasing="linear"
              content={<CustomTooltip />} 
              contentStyle={{
                backgroundColor: '#18181b',
                border: '1px solid #3f3f46',
                borderRadius: '8px',
                padding: '10px 14px',
                color: '#e4e4e7',
                fontSize: '13px',
                boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.3)',
              }}
            />

            {/* median line */}
            <Line
                  isAnimationActive={false}
              type="monotone"
              dataKey="values.median"
              name="median"
              stroke={linesMeta["median"]["stroke"]}
              strokeWidth={3.5}
              fill="none"
              dot={false}
              activeDot={{ 
                r: 6, 
                fill: linesMeta["median"]["stroke"],
                stroke: '#fff', 
                strokeWidth: 2 
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </ContentBlock>
  );
}
