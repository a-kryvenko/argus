export default function Help() {
    return (
        <div className="container color-default">
            <h2>Argus Sunwatch</h2>
            <br />
            <h3>Solar Activity Impact Forecasting & Decision Intelligence</h3>

            <p>
                <span>Github: </span> <a href="https://github.com/a-kryvenko/argus" target="_blank">https://github.com/a-kryvenko/argus</a>
            </p>
            <br />

            <h3>API</h3>
            <p>API documentation: <a href="/api/v1/docs">/api/v1/docs</a></p>
            <br />
            <br />
            <p>
                All models are trained on 2010-2024 observations data. Evaluated on 2025 year observations
            </p>
            <br />
            <br />
            <p>Contact email: <a href="mailto:krivenko.a.b@gmail.com">krivenko.a.b@gmail.com</a></p>
        </div>
    )
}