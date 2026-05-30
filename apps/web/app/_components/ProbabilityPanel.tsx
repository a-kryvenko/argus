import ProbabilityStrip from "./ProbabilityStrip";

export default function ProbabilityPanel({ data }: {data: any}) {
    return (
    <div style={{ marginTop: 30 }}>
      <h2>Solar Wind Exceedance Probability</h2>
      
      <div>
        <ProbabilityStrip
          data={data}
          label="> 500 km/s"
          keyName="prob_v_gt_500"
        />

        <ProbabilityStrip
          data={data}
          label="> 600 km/s"
          keyName="prob_v_gt_600"
        />

        <ProbabilityStrip
          data={data}
          label="> 700 km/s"
          keyName="prob_v_gt_700"
        />
      </div>
    </div>
  );
}