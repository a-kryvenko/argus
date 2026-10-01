"use client";
import ReactECharts from "echarts-for-react";
import type { DragPoint } from "./leo-drag";

export default function LeoDragChart({ rows }: { rows: DragPoint[] }) {
  return (
    <div role="img" aria-label="Cumulative altitude loss over time. Exact values are available in the hourly estimates table.">
      <ReactECharts style={{ height: 280, width: "100%" }} option={{
        animation: false,
        color: ["#79b9e6"],
        grid: { left: 65, right: 20, top: 35, bottom: 45 },
        tooltip: { trigger: "axis", confine: true, valueFormatter: (value: number) => `${value.toFixed(3)} m` },
        xAxis: { type: "value", name: "Hours from issue", nameLocation: "middle", nameGap: 28,
          min: 0, max: rows.at(-1)?.lead_hours, axisLabel: { color: "#a4b1bf" }, splitLine: { show: false } },
        yAxis: { type: "value", name: "Altitude loss (m)", axisLabel: { color: "#a4b1bf" },
          splitLine: { lineStyle: { color: "#26333f", type: "dashed" } } },
        series: [{ name: "Altitude loss", type: "line", showSymbol: false,
          areaStyle: { opacity: 0.1 }, data: rows.map(row => [row.lead_hours, row.estimated_altitude_loss_m]) }],
      }} />
    </div>
  );
}
