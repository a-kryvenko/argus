'use client';
import { useEffect, useRef } from 'react';
import { init, type ECharts } from 'echarts';
import type { EChartsReactProps } from 'echarts-for-react';

type Props = Pick<EChartsReactProps, 'option' | 'style' | 'notMerge' | 'onEvents'>;

export default function EChart({ option, style, notMerge = false, onEvents }: Props) {
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
      const listener = (params: unknown) => handler(params, instance);
      instance.on(event, listener);
      return { event, listener };
    });
    return () => {
      if (!instance.isDisposed()) listeners.forEach(({ event, listener }) => instance.off(event, listener));
    };
  }, [option, notMerge, onEvents]);
  return <div ref={element} data-chart style={{ height: 300, ...style }} />;
}
