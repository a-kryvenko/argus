export default function RiskStrip({ data, label, keyName }) {
  function getRiskColor(v) {
    if (v < 0.3) return "#2ecc71";   // green
    if (v < 0.6) return "#f1c40f";   // yellow
    if (v < 0.8) return "#e67e22";   // orange
    return "#e74c3c";                // red
  }
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ marginBottom: 5 }}>{label}</div>

      <div style={{ display: "flex", gap: 2 }}>
        {data.map((d, i) => {
          const value = d[keyName]; // 0–1

          const isNewDay =
            i > 0 &&
            new Date(data[i].time).getUTCDate() !==
                new Date(data[i - 1].time).getUTCDate();

          return (
            <div
              key={i}
              title={`${d.time} → ${Math.round(value * 100)}%`}
              style={{
                width: "1%",
                height: 30,
                background: getRiskColor(value),
                borderLeft: isNewDay ? "2px solid #fff" : "none"
              }}
            />
          );
        })}
      </div>
    </div>
  );
}