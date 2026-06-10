import {
  CartesianGrid,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart
} from "recharts";

export default function WindThresholdChart({ data }: {data: any}) {
  if (!data || data.length == 0) {
    return (
      <div>
        <h2>Wind Speed Forecast Metrics</h2>
        <p>Loading...</p>
      </div>
    );
  }
  return (
    <div style={{ width: "100%"}}>
      <div style={{ paddingLeft: 70}}>
        <h2>Threshold Event Metrics</h2>
        <h3>ROC AUC per threshold bar per lead hour</h3>
      </div>

      <div style={{ height: 400 }}>
        <ResponsiveContainer>
          <LineChart data={data} syncId="metricsChartSync">

            <XAxis
              dataKey="lead_hours"
            />
            <YAxis
              type="number"
              domain={["dataMin", "dataMax"]}
              ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
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
            />

            <Line
              type="monotone"
              dataKey="roc_auc_500"
              stroke="#3bafed"
              strokeWidth={3.5}
              fill="none"
              dot={false}
              activeDot={{ 
                r: 6, 
                fill: '#3bafed',
                stroke: '#fff', 
                strokeWidth: 2 
              }}
            />

            <Line
              type="monotone"
              dataKey="roc_auc_600"
              stroke="#eba93e"
              strokeWidth={3.5}
              fill="none"
              dot={false}
              activeDot={{ 
                r: 6, 
                fill: '#eba93e',
                stroke: '#fff', 
                strokeWidth: 2 
              }}
            />

            <Line
              type="monotone"
              dataKey="roc_auc_700"
              stroke="#e23719"
              strokeWidth={3.5}
              fill="none"
              dot={false}
              activeDot={{ 
                r: 6, 
                fill: '#e23719',
                stroke: '#fff', 
                strokeWidth: 2 
              }}
            />

          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}