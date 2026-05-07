'use client';

import WindChart from "./components/WindChart";
import { prepareChartData } from "./utils/transform";
import RiskStrip from "./components/RiskStrip"
import ProbabilityPanel from "./components/ProbabilityPanel";
import { forecast } from "./data/mockData";

function App() {
  const data = prepareChartData(forecast);

  return (
    <div style={{ padding: 20 }}>
      <h1>Argus Sunwatch</h1>

      <WindChart data={data} />

      <h2>Risk Overview</h2>

      <RiskStrip
        data={forecast}
        label="KP Risk"
        keyName="kp_risk_proxy"
      />

      <RiskStrip
        data={forecast}
        label="Satellite Drag Risk"
        keyName="drag_risk_proxy"
      />

      <ProbabilityPanel data={forecast} />

    </div>
  );
}

export default App;