export default function Help() {
    return (
        <div className="container">
            <h2>Argus Sunwatch</h2>
            <h3>Solar Activity Impact Forecasting & Decision Intelligence</h3>

            <p>Here You can find probabilistic wind speed <a href="/">forecast</a>, <a href="/api/forecast">API</a></p>

            <h3>Pages</h3>
            <ul>
                <li><a href="/">Solar Wind Forecast</a></li>
                {/* <li><a href="/metrics">Metrics</a></li> */}
            </ul>

            <h3>API points</h3>
            <ul>
                <li>/api/forecast/wind-speed</li>
                <li>/api/forecast/threshold</li>
                <li>/api/forecast/all</li>
            </ul>
            <p>More API info on <a href="/api/docs">/api/docs</a></p>
        </div>
    )
}