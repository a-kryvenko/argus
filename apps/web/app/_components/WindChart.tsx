import {
  CartesianGrid,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart
} from "recharts";

import get_lead_datetime from "../_utils/date_time";
import "./windchart.css"

const xAxisMeta = get_lead_datetime();

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

export default function WindChart({ data }: {data: any}) {
  if (!data || data.length == 0) {
    return (
      <div>
        <h2>Solar Wind Speed</h2>
        <p>Loading...</p>
      </div>
    );
  }

  const rechartsData = xAxisMeta.map((x, i) => ({
    ...x,
    xKey: x.timestamp,
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
          {row.dayName}, {row.hour}
        </div>

        {payload.map((item: {dataKey: any, name: string, value: Number}) => (
          <div
            className="tooltip__row"
            key={item.dataKey}
            style={{
              color: linesMeta[item.name]["color"],
              fontWeight: linesMeta[item.name]["fontWeight"]
            }}
          >
            <span>{linesMeta[item.name]["name"]}</span>
            <span>{item.value.toString()} km/s</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div style={{ width: "100%"}}>
      <h2>Solar Wind Speed</h2>

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
              dataKey="index"
              type="number"
              domain={[0, 95]}
              ticks={rechartsData
                .filter((_, i) => i % 6 === 0)
                .map((d) => d.index)}
              tickFormatter={(index) => {
                return rechartsData[index]?.hour ?? "";
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
              tick={{ fontSize: 12 }}
            />

            <CartesianGrid strokeDasharray="3 3" stroke="#212121" />

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

            {/* invisible base */}
            <Line
              type="monotone"
              dataKey="values.low"
              name="low"
              stroke={linesMeta["low"]["stroke"]}
              fill="none"
              dot={false}
              activeDot={false}
            />

            {/* invisible base */}
            <Line
              type="monotone"
              dataKey="values.high"
              name="high"
              stroke={linesMeta["high"]["stroke"]}
              fill="none"
              dot={false}
              activeDot={false}
            />

          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}