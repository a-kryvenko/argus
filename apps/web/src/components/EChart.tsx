'use client';
import { useEffect, useRef, type CSSProperties } from 'react';
import { init, type ECharts } from 'echarts';

type Props<Event> = {
  option: Parameters<ECharts['setOption']>[0];
  style?: CSSProperties;
  notMerge?: boolean;
  onEvents?: Record<string, (event: Event, chart: ECharts) => void>;
};

export default function EChart<Event>({ option, style, notMerge = false, onEvents }: Props<Event>) {
  const element = useRef<HTMLDivElement>(null);
  const chart = useRef<ECharts | null>(null);
  useEffect(() => {
    const instance = init(element.current!);
    chart.current = instance;
    const observer = new ResizeObserver(() => {
      if (!instance.isDisposed()) instance.resize();
    });
    observer.observe(element.current!);
    return () => {
      observer.disconnect();
      chart.current = null;
      instance.dispose();
    };
  }, []);
  useEffect(() => {
    const instance = chart.current!;
    instance.setOption(option, { notMerge });
    const listeners = Object.entries(onEvents ?? {}).map(([event, handler]) => {
      const listener = (params: unknown) => handler(params as Event, instance);
      instance.on(event, listener);
      return { event, listener };
    });
    return () => {
      if (!instance.isDisposed()) listeners.forEach(({ event, listener }) => instance.off(event, listener));
    };
  }, [option, notMerge, onEvents]);
  return <div ref={element} data-chart style={{ height: 300, ...style }} />;
}
