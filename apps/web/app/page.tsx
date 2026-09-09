import ForecastOverview from "./_components/ForecastOverview";
export default function Forecast() {
  return (
    <main className="container">
      <h1>Solar wind and geomagnetic forecasts</h1>
      <p className="product-description">
        Forecast probabilities and solar wind speed. All times in UTC.
      </p>
      <ForecastOverview />
    </main>
  );
}
