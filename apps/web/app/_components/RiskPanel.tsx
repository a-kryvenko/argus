import RiskStrip from "./RiskStrip";

export default function RiskPanel({ data }: {data: any})
{
    return (
        <div>
            <h2>Risk Overview</h2>
            <div>
                <RiskStrip
                    data={data}
                    label="KP Risk"
                    keyName="kp_risk"
                />

                {/* <RiskStrip
                    data={data}
                    label="Satellite Drag Risk"
                    keyName="drag_risk_proxy"
                /> */}
            </div>
        </div>
    );
}