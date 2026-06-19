import ReactECharts from "echarts-for-react";
import get_lead_datetime from "../_utils/date_time"

const base = new Date();
base.setMinutes(0, 0, 0);
base.setHours(base.getHours() + 1);

const xAxisMeta = get_lead_datetime();

export default function HeatMap({title, yLabels, data}: {title: string, yLabels: Array<string>, data: Array<Array<Number>>}) {
  const option = {
    tooltip: {
      formatter: (params: any) => {
        const [xIndex, yIndex, value] = params.data;
        const x = xAxisMeta[xIndex];

        return `
          ${x.dayName}, ${x.hour}<br/>
          ${yLabels[yIndex]}: <b>${value}%</b> risk
        `;
      },
    },

    grid: {
      top: 0,
      right: 0,
      bottom: 0,
      left: 70,
    },

    xAxis: {
      type: "category",
      data: xAxisMeta.map(x => x.hour),
      splitArea: { show: false },
      axisLabel: {
        interval: 11,
        formatter: (value: string) => value,
      },
    },

    yAxis: {
      type: "category",
      data: yLabels,
      splitArea: { show: false },
      axisLabel: {
        width: 58,
        overflow: "truncate",
        align: "right",
      },
    },

    visualMap: {
      min: 0,
      max: 100,
      calculable: false,
      show: false,
      text: ["High risk", "Low risk"],
      formatter: "{value}%",
      type: "continuous",
      orient: "horizontal",
      left: "center",
      bottom: 8,
      inRange: {
        color: ['#dfe6e9', '#7480ff', '#4a09e3', '#ee7f7f', '#d63031']
      },
    },

    series: [
      {
        name: { title },
        type: "heatmap",
        data: data,
        itemStyle: {
          borderColor: "#fff",
          borderWidth: 1,
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 0,
            shadowColor: "transparent",
          },
        },
      },
    ],
  };

  return (
    <>
    <h2>{ title }</h2>
    <ReactECharts
      option={option}
      style={{ height: 30 * yLabels.length, width: "100%" }}
      notMerge
      lazyUpdate
    />
    </>
  );
}