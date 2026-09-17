"use client";
import ReactECharts from "echarts-for-react";
import type { Stats } from "./ApiActivity";

export default function TrafficChart({
  rows,
  label,
}: {
  rows: Stats["hours"];
  label?: string;
}) {
  return (
    <div
      role="img"
      aria-label={
        label ??
        "Hourly API request volume and errors. Exact values are available in the API statistics hourly activity table."
      }
    >
      <ReactECharts
        style={{ height: 280, width: "100%" }}
        option={{
          animation: false,
          useUTC: true,
          color: ["#a78bfa", "#fb7185"],
          grid: { left: 48, right: 20, top: 38, bottom: 38 },
          legend: {
            top: 0,
            right: 8,
            icon: "circle",
            itemWidth: 7,
            itemHeight: 7,
            textStyle: { color: "#a1a1aa", fontSize: 11 },
          },
          tooltip: {
            trigger: "axis",
            confine: true,
            backgroundColor: "#202023",
            borderColor: "#353539",
            textStyle: { color: "#f4f4f5", fontSize: 12 },
          },
          xAxis: {
            type: "time",
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: { color: "#71717a", fontSize: 10, hideOverlap: true },
            splitLine: { show: false },
          },
          yAxis: {
            type: "value",
            minInterval: 1,
            axisLabel: { color: "#71717a", fontSize: 10 },
            splitLine: { lineStyle: { color: "#27272a", type: "dashed" } },
          },
          series: [
            {
              name: "Requests",
              type: "line",
              showSymbol: rows.length === 1,
              symbolSize: 6,
              lineStyle: { width: 2 },
              areaStyle: {
                color: {
                  type: "linear",
                  x: 0,
                  y: 0,
                  x2: 0,
                  y2: 1,
                  colorStops: [
                    { offset: 0, color: "rgba(167,139,250,.22)" },
                    { offset: 1, color: "rgba(167,139,250,.01)" },
                  ],
                },
              },
              data: rows.map((row) => [row.hour, row.requests]),
            },
            {
              name: "Errors (4xx + 5xx)",
              type: "line",
              showSymbol: rows.length === 1,
              symbolSize: 5,
              lineStyle: { width: 1.5 },
              data: rows.map((row) => [
                row.hour,
                row.errors_4xx + row.errors_5xx,
              ]),
            },
          ],
        }}
      />
    </div>
  );
}
