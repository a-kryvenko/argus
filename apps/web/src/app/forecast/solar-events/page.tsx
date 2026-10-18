import WorkspaceShell from '../../../components/WorkspaceShell';
import SolarEvents from './SolarEvents';

export const metadata = { title: 'Solar events', description: 'CME arrival times and expected solar-wind speed at Earth.' };

export default function SolarEventsPage() {
  return <WorkspaceShell section="Solar events"><SolarEvents /></WorkspaceShell>;
}
