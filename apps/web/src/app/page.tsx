import type { Metadata } from 'next';
import ObservationOverview from '../components/ObservationOverview';

export const metadata: Metadata = { title: 'Solar Weather overview | Argus Sunwatch' };

export default function Home() {
  return <ObservationOverview />;
}
