import ProbabilityStrip from "./ProbabilityStrip";

export default function ProbabilityPanel({ data }: {data: Array<any>}) {
    return (
    <div style={{ marginTop: 30 }}>
      <h2>Solar Wind Exceedance Probability</h2>
      
      <div>
        <ProbabilityStrip
          data={data}
          label="> 450 km/s"
          keyName="p_450"
        />

        <ProbabilityStrip
          data={data}
          label=">= 500 km/s"
          keyName="p_500"
        />

        <ProbabilityStrip
          data={data}
          label=">= 600 km/s"
          keyName="p_600"
        />
      </div>
    </div>
  );
}