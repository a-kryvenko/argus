import RiskStrip from "./RiskStrip";

export default function RiskPanel({ data })
{
    return (
        <div>
            <h2>Risk Overview</h2>
            {data && data.length > 0 ? (
                <div style={{padding: "0 0 0 70px"}}>
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
            ): (
                <p>Loading...</p>
            )}
        </div>
    );
}