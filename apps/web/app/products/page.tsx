export default function Products()
{
    return (
        <div className="container color-default">
            <ul>
                <li><a href="/products/solar-wind">Solar Wind</a></li>
                <li><a href="/products/geomagnetic">Kp, Ap, GIC Risk</a></li>
                <li><a href="/products/hmf">Heliospheric Magnetic Field</a></li>
                {/* <li><a href="/products/satelites">Satelites Risk</a></li> */}
                {/* <li><a href="/products/power-grid">Power Grid Risk</a></li> */}
            </ul>
        </div>
    )
}