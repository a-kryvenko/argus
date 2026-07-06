export default function Help() {
    return (
        <div className="container color-default">
            <h2>Argus Sunwatch</h2>
            <br />
            <h3>Solar Activity Impact Forecasting & Decision Intelligence</h3>

            <p>Here You can find probabilistic Solar Activity <a href="/products">forecast</a>, <a href="/api/v1/docs">API</a> and <a href="/metrics">metrics</a>.</p>

            <p>
                <span>Github: </span> <a href="https://github.com/a-kryvenko/argus">https://github.com/a-kryvenko/argus</a>
            </p>
            <br />

            <h3>API</h3>
            <p>API documentation: <a href="/api/v1/docs">/api/v1/docs</a></p>
            <br />
            <br />
            <table>
                <tbody>
                    <tr>
                        <th></th>
                        <th>Training period</th>
                        <th>Evaluation period</th>
                    </tr>
                    <tr>
                        <td>Plasma</td>
                        <td>2010-2024</td>
                        <td>2025</td>
                    </tr>
                    <tr>
                        <td>HMF</td>
                        <td>2010-2024</td>
                        <td>2025</td>
                    </tr>
                </tbody>
            </table>
            <br />
            <br />
            <p>Contact email: <a href="mailto:krivenko.a.b@gmail.com">krivenko.a.b@gmail.com</a></p>
        </div>
    )
}