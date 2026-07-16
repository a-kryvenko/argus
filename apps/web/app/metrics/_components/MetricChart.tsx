import {
  CartesianGrid,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart
} from "recharts";

import "../../_components/charts.css"
import ContentBlock from "../../_components/ContentBlock";

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

export default function MetricChart({ data, title, labels }: {data: Array<any>, title: string, labels: Record<string, string>}) {
  if (!data || data.length == 0) {
    return (
      <div>
        <h3>{ title }</h3>
        <p>Loading...</p>
      </div>
    );
  }

  const leadHours = data.length;

  const xAxisMeta = Array.from({length: leadHours}, (_, i) => i + 1);

  const rechartsData = xAxisMeta.map((x, i) => ({
    x,
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
          Forecast horizon, hours: <b>{row.x}</b>
        </div>
        <div>
          {Object.entries(labels).map(([key, value], i) => (
            <div
              className="tooltip__row"
              key={key}
              style={{
                color: linesMeta[i].color,
                fontWeight: linesMeta[i].fontWeight,
              }}
            >
              <span>{ value }</span>
              <span>{row.values[key]}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <ContentBlock>
      <h3 className="heading">{ title }</h3>

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
              dataKey="x"
              type="number"
              domain={[1, leadHours + 1]}
            />
            
            <YAxis
              type="number"
              tickCount={6}
              width={70}
              tick={{ fontSize: 12 }}
              domain={[
                (min: number) => Math.floor(min * 10) / 10,
                (max: number) => Math.ceil(max * 10) / 10,
              ]}
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

            {Object.entries(labels).map(([key, value], i) => {
              const meta = linesMeta[i];
          
              return (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={`values.${key}`}
                  name={key}
                  stroke={meta["stroke"]}
                  strokeWidth={3.5}
                  fill="none"
                  dot={false}
                  activeDot={{ 
                    r: 6, 
                    fill: meta["stroke"],
                    stroke: '#fff', 
                    strokeWidth: 2 
                  }}
                />
              );
            })}

            <Legend formatter={(value) => labels[value] ?? value} />

          </LineChart>
        </ResponsiveContainer>
      </div>
    </ContentBlock>
  );
}