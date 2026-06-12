function getColor(p: number) {
  if (p < 0.2) return "#dfe6e9";
  if (p < 0.4) return "#74b9ff";
  if (p < 0.6) return "#0984e3";
  if (p < 0.8) return "#6c5ce7";
  return "#d63031";
}

export default function ProbabilityStrip({ data, label, keyName }: {data: Array<any>, label: string, keyName: string}) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ fontSize: 12, marginBottom: 6 }}>
        {label}
      </div>

      {data && data.length > 0 ? (
        <div style={{ display: "flex", gap: 2 }}>
          {data.map((d, i) => {
            const p = d[keyName] * 100; // convert to %

            return (
              <div
                key={i}
                title={`${d.time} - ${p.toFixed(0)}%`}
                style={{
                  width: "1%",
                  height: 18,
                  background: getColor(p / 100)
                }}
              />
            );
          })}
        </div>
      ) : (
        <div style={{ height: 18, width: "100%", backgroundColor: "#313131" }}></div>
      )}
    </div>
  );
}