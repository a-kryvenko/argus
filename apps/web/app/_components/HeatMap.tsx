'use client';
import ReactECharts from 'echarts-for-react';
import { formatForecastTime } from '../_utils/forecast';
import styles from './forecast.module.css';

export default function HeatMap({ title, yLabels, data, times, selectedTime, onSelectTime }: {
  title: string; yLabels: string[]; data: number[][]; times: string[];
  selectedTime?: string; onSelectTime?: (time: string) => void;
}) {
  const option = {
    animation: false,
    aria: { enabled: true, description: `${title}. Threshold probabilities from 0 to 100 percent. Empty cells indicate unavailable data. All times UTC. Exact values are available in the forecast inspector and hourly table.` },
    tooltip: {
      confine: true, transitionDuration: 0, backgroundColor: '#1b2632', borderColor: '#3c5062', textStyle: { color: '#e4edf5', fontSize: 12 },
      formatter: (params: { data: number[] }) => {
        const [x, y, value] = params.data;
        return `${formatForecastTime(times[x])}<br/>${yLabels[y]}: <b>${value}%</b> probability`;
      },
    },
    grid: { top: 8, right: 16, bottom: 45, left: 12, containLabel: true },
    xAxis: { type: 'category', data: times, axisTick: { show: false }, axisLine: { show: false },
      axisLabel: { color: '#92a4b5', fontSize: 10, fontFamily: 'monospace', hideOverlap: true,
        formatter: (value: string) => `${value.slice(5, 10)}\n${value.slice(11, 16)}` } },
    yAxis: { type: 'category', data: yLabels, axisTick: { show: false }, axisLine: { show: false }, axisLabel: { color: '#b4c3d1', fontSize: 10 } },
    visualMap: { min: 0, max: 100, show: false, inRange: { color: ['#1b2a37', '#254962', '#377997', '#8bbfca', '#e0be77'] } },
    series: [{ name: title, type: 'heatmap', data, itemStyle: { borderColor: '#131b24', borderWidth: 1 },
      emphasis: { itemStyle: { borderColor: '#e0edf7', borderWidth: 1 } },
      markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: '#b9c9d7', type: 'dashed', width: 1 },
        data: selectedTime && times.includes(selectedTime) ? [{ xAxis: selectedTime }] : [] },
    }],
  };
  return <section className={styles.chart} aria-label={title}>
    <div className={styles.chartHeading}><h3>{title}</h3><span className={styles.chartTag}>PROBABILITY</span></div>
    {data.length ? <>
      <ReactECharts option={option} style={{ height: 42 * yLabels.length + 70, width: '100%' }} notMerge
        onEvents={{ click: (event: { data?: number[]; componentType?: string }) => {
          if (event.componentType === 'series' && event.data && times[event.data[0]]) onSelectTime?.(times[event.data[0]]);
        } }} />
      <div className={styles.probabilityLegend} aria-label="Probability color scale: 0 to 100 percent"><span>Probability</span><span>0%</span><i aria-hidden="true" /><span>100%</span><span>· UTC</span></div>
      <p className={styles.chartNote}>Per-time probability of meeting or exceeding each threshold. Click a cell to inspect. Empty cells are unavailable.</p>
    </> : <p className={styles.emptyChart} role="status">No probability forecast available in this horizon.</p>}
  </section>;
}
