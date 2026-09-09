import ForecastOverview from "./_components/ForecastOverview";
import Link from "next/link";
import styles from "./page.module.css";
export default function Forecast() {
  return (
    <main className="container">
      <h1>Solar wind and geomagnetic forecasts</h1>
      <div className={styles.intro}>
        <p className="product-description">
          Explore the probability of elevated geomagnetic activity and faster
          solar wind. Each cell shows the chance of reaching or exceeding a
          threshold at that hour. All times in UTC.
        </p>
        <nav className={styles.links} aria-label="Forecast resources">
          <Link href="/live">Live observations →</Link>
          <Link href="/metrics">Model performance →</Link>
        </nav>
      </div>
      <ForecastOverview />
    </main>
  );
}
