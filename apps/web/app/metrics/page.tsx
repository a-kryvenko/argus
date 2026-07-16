export default function Metrics()
{
    return (
        <div className="container color-default">
            <h1 className="text-center">Metrics</h1>
            <ul>
                <li><a href="/metrics/solar-wind">Solar Wind Speed</a></li>
                <li><a href="/metrics/kp">Kp</a></li>
                <li><a href="/metrics/hmf">Heliospheric Magnetic Field</a></li>
                <li><a href="/metrics/southward-bz">Southward Bz</a></li>
                {/* <li><a href="/products/satelites">Satelites Risk</a></li> */}
                {/* <li><a href="/products/power-grid">Power Grid Risk</a></li> */}
            </ul>
        </div>
    )
}