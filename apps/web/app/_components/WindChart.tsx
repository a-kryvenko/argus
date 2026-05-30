import {
  CartesianGrid,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart
} from "recharts";

export default function WindChart({ data }: {data: any}) {
  if (!data || data.length == 0) {
    return (
      <div>
        <h2>Solar Wind Speed</h2>
        <p>Loading...</p>
      </div>
    );
  }
  return (
    <div style={{ width: "100%"}}>
      <div style={{ paddingLeft: 70}}>
        <h2>Solar Wind Speed</h2>
      </div>

      <div style={{ height: 400 }}>
        <ResponsiveContainer>
          <LineChart data={data}>

            <XAxis
              dataKey="time"
              tickFormatter={(t) =>
                new Date(t).toISOString().slice(11, 16) // HH:mm UTC
              }
            />
            <YAxis
              type="number"
              domain={[
                dataMin => Math.floor(dataMin / 10) * 10,
                dataMax => Math.ceil(dataMax / 10) * 10,
              ]}
              tickCount={6}
            />

            <CartesianGrid strokeDasharray="3 3" stroke="#212121" />

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
            <Line
              type="monotone"
              dataKey="low"
              stroke="none"
              fill="transparent"
              dot={false}
              activeDot={false}
            />

            {/* uncertainty band */}
            {/* <Area
              type="monotone"
              dataKey="range"
              stackId="1"
              stroke="none"
              fill="#8884d8"
              fillOpacity={0.3}
            /> */}

            {/* median line */}
            <Line
              type="monotone"
              dataKey="median"
              stroke="#f97316"
              strokeWidth={3.5}
              fill="none"
              dot={false}
              activeDot={{ 
                r: 6, 
                fill: '#f97316',
                stroke: '#fff', 
                strokeWidth: 2 
              }}
            />

            {/* invisible base */}
            <Line
              type="monotone"
              dataKey="high"
              stroke="none"
              fill="transparent"
              dot={false}
              activeDot={false}
            />

          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}