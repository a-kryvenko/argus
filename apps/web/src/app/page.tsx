import type { Metadata } from 'next';
import ObservationOverview from '../components/ObservationOverview';

export const metadata: Metadata = { title: 'Overview' };

export default function Home() {
  return <ObservationOverview />;
}
