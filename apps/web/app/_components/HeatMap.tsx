import ReactECharts from "echarts-for-react";
import { formatForecastTime } from "../_utils/forecast";
import "./charts.css";
import ContentBlock from "./ContentBlock";

export default function HeatMap({
  title,
  yLabels,
  data,
  times,
}: {
  title: string;
  yLabels: string[];
  data: number[][];
  times: string[];
}) {
  const option = {
    aria: {
      enabled: true,
      description: `${title}. Color indicates threshold probability from 0 to 100 percent. Times are UTC.`,
    },
    tooltip: {
      confine: true,
      backgroundColor: "#21172e",
      borderColor: "#675579",
      textStyle: { color: "#f7f3fc" },
      formatter: (params: { data: number[] }) => {
        const [x, y, value] = params.data;
        return `${formatForecastTime(times[x])}<br/>${yLabels[y]}: <b>${value}%</b> probability`;
      },
    },
    grid: { top: 8, right: 12, bottom: 48, left: 8, containLabel: true },
    xAxis: {
      type: "category",
      data: times,
      axisLabel: {
        color: "#b9aec7",
        hideOverlap: true,
        formatter: (value: string) => {
          const d = new Date(value);
          return `${d.getUTCDate()} ${d.toLocaleString("en-GB", { month: "short", timeZone: "UTC" })}\n${d.toISOString().slice(11, 16)}`;
        },
      },
    },
    yAxis: {
      type: "category",
      data: yLabels,
      axisLabel: { color: "#d8cfe3", fontSize: 12 },
    },
    visualMap: {
      min: 0,
      max: 100,
      show: false,
      inRange: {
        color: ["#272138", "#6655b8", "#a78bfa", "#f5a35b", "#f65c71"],
      },
    },
    series: [
      {
        name: title,
        type: "heatmap",
        data,
        itemStyle: { borderColor: "#160f20", borderWidth: 1 },
      },
    ],
  };
  return (
    <ContentBlock>
      <h2>{title}</h2>
      {data.length ? (
        <>
          <ReactECharts
            option={option}
            style={{ height: 46 * yLabels.length + 78, width: "100%" }}
            notMerge
          />
          <div
            className="probability-legend"
            aria-label="Probability color scale: 0 to 100 percent"
          >
            <span>Probability</span>
            <span>0%</span>
            <span className="probability-legend__scale" aria-hidden="true" />
            <span>100%</span>
            <span>· UTC</span>
          </div>
        </>
      ) : (
        <p className="forecast-meta" role="status">
          No probability forecast available.
        </p>
      )}
    </ContentBlock>
  );
}
