import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from "recharts";

export default function WindChart({ data }) {
  if (!data || data.length == 0) {
    return (
      <div>
        <h2>Solar Wind Speed</h2>
        <p>Loading...</p>
      </div>
    );
  }
  return (
    <div style={{ width: "100%", height: 400 }}>
      <h2>Solar Wind Speed</h2>

      <ResponsiveContainer>
        <AreaChart data={data}>
          
          <XAxis
            dataKey="time"
            tickFormatter={(t) =>
              new Date(t).toISOString().slice(11, 16) // HH:mm UTC
            }
          />

          <YAxis
            type="number"
            domain={[200, 650]}
            allowDataOverflow={true}
          />

          <Tooltip
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
            labelStyle={{ color: '#a1a1aa', fontSize: '12px' }}
            itemStyle={{ color: '#f97316' }}
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
            stroke="#f97316"
            strokeWidth={3.5}
            fill="none"
            activeDot={{ 
              r: 6, 
              fill: '#f97316',
              stroke: '#fff', 
              strokeWidth: 2 
            }}
          />

        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}