export default function Help() {
    return (
        <div className="container color-default">
            <h2>Argus Sunwatch</h2>
            <h3>Solar Activity Impact Forecasting & Decision Intelligence</h3>

            <p>Here You can find probabilistic wind speed <a href="/">forecast</a>, <a href="/api/docs">API</a> and <a href="/metrics">metrics</a>.</p>

            <p>
                <span>Github: </span> <a href="https://github.com/a-kryvenko/argus">https://github.com/a-kryvenko/argus</a>
            </p>

            <h3>API points</h3>
            <ul>
                <li><a href="/api/forecast/wind-speed/">/api/forecast/wind-speed</a></li>
                <li><a href="/api/forecast/threshold">/api/forecast/threshold</a></li>
                <li><a href="/api/forecast/all">/api/forecast/all</a></li>
                <li><a href="/api/metrics/wind-speed">/api/metrics/wind-speed</a></li>
                <li><a href="/api/metrics/threshold">/api/metrics/threshold</a></li>
                <li><a href="/api/metrics/all">/api/metrics/all</a></li>
            </ul>
            <p>More API info on <a href="/api/docs">/api/docs</a></p>
        </div>
    )
}