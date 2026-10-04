import { metricNumber, metricTitle } from '../_utils/transform';

export default function MetricValue({ value }: { value: number | null | undefined }) {
  return <span title={metricTitle(value)}>{metricNumber(value)}</span>;
}
