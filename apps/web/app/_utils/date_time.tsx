export default function get_lead_datetime(): Array<any>
{
    const base = new Date();
    base.setMinutes(0, 0, 0);
    base.setHours(base.getHours() + 1);

    const xAxisMeta = Array.from({ length: 96 }, (_, i) => {
        const d = new Date(base);
        d.setHours(base.getHours() + i);

        return {
            index: i,
            timestamp: d.valueOf(),
            hour: d.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
                hour12: false,
            }),
            dayName: d.toLocaleDateString([], {
                weekday: "long",
            }),
            date: d,
        };
    });
    return xAxisMeta;
}