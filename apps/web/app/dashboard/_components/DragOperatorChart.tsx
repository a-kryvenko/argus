'use client';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption, LineSeriesOption } from 'echarts';
import { number, utc } from './leo-drag';
import type { OperatorChart } from './drag-monitor';

const colors = ['#79b9e6', '#c4a4f5', '#64d8b0', '#f4bd72', '#e78fa5'];

export default function DragOperatorChart({ chart, showPrevious }: { chart: OperatorChart; showPrevious: boolean }) {
  const series: LineSeriesOption[] = [];
  for (const [index, mode] of chart.modes.entries()) {
    const name = mode.mode + (mode.active ? ' · Active' : '');
    const color = colors[index % colors.length];
    if (mode.error) continue;
    for (const [phase, points] of [['elapsed', mode.elapsed], ['remaining', mode.remaining]] as const) {
      series.push({ name, type: 'line', showSymbol: false, clip: true,
        data: points.map(point => [point.lead_hours, point.altitude_loss_m]),
        lineStyle: { color, width: mode.active ? 3 : 2, opacity: phase === 'elapsed' ? 0.45 : 1 },
        itemStyle: { color },
        markPoint: phase === (mode.remaining.length ? 'remaining' : 'elapsed') ? { symbol: 'diamond', symbolSize: 11,
          label: { show: false }, data: mode.crossings.map(crossing => ({
            name: `${mode.mode}: ${crossing.level === 'high' ? 'High' : 'Review'} limit · ${utc(crossing.valid_time)} UTC`,
            coord: [crossing.lead_hours, crossing.altitude_loss_m],
            itemStyle: { color: crossing.level === 'high' ? '#f48c8c' : '#f4bd72' },
          })) } : undefined,
      });
    }
    if (showPrevious && mode.previous.length > 0) series.push({ name: `${mode.mode} · Previous`, type: 'line', showSymbol: false,
      data: mode.previous.map(point => [point.lead_hours, point.altitude_loss_m]),
      lineStyle: { color, type: 'dashed', width: 1.5, opacity: 0.65 }, itemStyle: { color } });
  }
  // This transparent reference series renders backend-supplied limits and times.
  series.push({ name: 'Limits', type: 'line', data: [], silent: true,
    markLine: { symbol: 'none', label: { color: '#dbe5ef', fontSize: 10, position: 'insideEndTop' }, data: [
      { name: 'Review', yAxis: chart.thresholds.elevated_altitude_loss_m, lineStyle: { color: '#f4bd72', type: 'dashed' }, label: { formatter: 'Review limit' } },
      { name: 'High', yAxis: chart.thresholds.high_altitude_loss_m, lineStyle: { color: '#f48c8c', type: 'dashed' }, label: { formatter: 'High limit' } },
      ...(chart.now_position === 'within' && chart.now_lead_hours !== null ? [{ name: 'Now', xAxis: chart.now_lead_hours,
        lineStyle: { color: '#e7eef5', type: 'solid' as const }, label: { formatter: 'Now', position: 'insideEndBottom' as const } }] : []),
    ] },
    markArea: chart.elapsed_until_hours !== null && chart.elapsed_until_hours > 0 ? {
      silent: true, itemStyle: { color: 'rgba(150, 165, 180, 0.08)' },
      data: [[{ xAxis: 0 }, { xAxis: chart.elapsed_until_hours }]],
    } : undefined,
  });
  const option: EChartsOption = {
    animation: false, aria: { enabled: true },
    legend: { type: 'scroll', top: 0, textStyle: { color: '#b5c3d1' },
      data: chart.modes.filter(mode => !mode.error).flatMap(mode => [mode.mode + (mode.active ? ' · Active' : ''),
        ...(showPrevious && mode.previous.length ? [`${mode.mode} · Previous`] : [])]) },
    grid: { left: 65, right: 24, top: 65, bottom: 48 },
    tooltip: { trigger: 'axis', confine: true, renderMode: 'richText',
      valueFormatter: value => `${number(Number(Array.isArray(value) ? value.at(-1) : value))} m` },
    xAxis: { type: 'value', min: 0, max: chart.horizon_hours, name: 'Hours from selected release',
      nameLocation: 'middle', nameGap: 30, axisLabel: { color: '#a4b1bf' }, splitLine: { show: false } },
    yAxis: { type: 'value', min: 0, max: chart.max_loss_m, name: 'Altitude loss (m)', axisLabel: { color: '#a4b1bf' },
      splitLine: { lineStyle: { color: '#26333f', type: 'dashed' } } },
    series,
  };
  return <div className="min-w-0 overflow-hidden" role="img" aria-label="Cumulative altitude loss for attitude modes, operator limits and current time. Exact values are available in chart data.">
    <ReactECharts style={{ height: 360, width: '100%' }} option={option} notMerge />
  </div>;
}
