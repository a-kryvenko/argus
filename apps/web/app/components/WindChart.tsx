import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from "recharts";

export default function WindChart({ data }) {
  return (
    <div style={{ width: "100%", height: 400 }}>
      <ResponsiveContainer>
        <AreaChart data={data}>
          
          <XAxis
            dataKey="time"
            tickFormatter={(t) =>
              new Date(t).toISOString().slice(11, 16) // HH:mm UTC
            }
          />

          <YAxis />

          <Tooltip
            labelFormatter={(label) =>
              new Date(label).toISOString()
            }
          />

          {/* invisible base */}
          <Area
            type="monotone"
            dataKey="low"
            stackId="1"
            stroke="none"
            fill="transparent"
          />

          {/* uncertainty band */}
          <Area
            type="monotone"
            dataKey="range"
            stackId="1"
            stroke="none"
            fill="#8884d8"
            fillOpacity={0.3}
          />

          {/* median line */}
          <Area
            type="monotone"
            dataKey="median"
            stroke="#000"
            fill="none"
          />

        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}