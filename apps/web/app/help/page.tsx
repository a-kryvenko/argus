import type { Metadata } from "next";
import Link from "next/link";
import { products } from "../_config/products";
import styles from "./page.module.css";

export const metadata: Metadata = {
  title: "Help | Argus Sunwatch",
  description: "Learn how to read Argus Sunwatch forecasts, interpret probabilities and metrics, and access observation and forecast data.",
};

const sections = [
  ["start", "Getting started"],
  ["forecasts", "Reading forecasts"],
  ["products", "Forecast products"],
  ["observations", "Live data"],
  ["metrics", "Model performance"],
  ["api", "API & support"],
] as const;

export default function Help() {
  return (
    <main className={`container ${styles.page}`}>
      <header className={styles.hero}>
        <p className={styles.eyebrow}>ARGUS SUNWATCH · HELP</p>
        <h1>Understand the forecast.</h1>
        <p className={styles.intro}>
          Argus Sunwatch brings solar wind and geomagnetic forecasts together with
          recent observations. Use this guide to read the charts, explore uncertainty,
          and check how the models performed on historical data.
        </p>
        <div className={styles.actions}>
          <Link href="/">Explore the forecast <span aria-hidden="true">→</span></Link>
          <Link href="/live">Check live observations <span aria-hidden="true">→</span></Link>
        </div>
      </header>

      <div className={styles.layout}>
        <nav className={styles.contents} aria-label="Help topics">
          <p>ON THIS PAGE</p>
          {sections.map(([id, label]) => <a href={`#${id}`} key={id}>{label}</a>)}
        </nav>

        <div className={styles.content}>
          <section id="start" className={styles.section}>
            <h2>Getting started</h2>
            <ol className={styles.steps}>
              <li><strong>Check current conditions.</strong> Open <Link href="/live">Live</Link> and check the observation timestamp to see how recent the available data is.</li>
              <li><strong>Choose a forecast.</strong> Browse <Link href="/products">Products</Link> for the quantity you want to follow. Each product has its own charts, units, and API links.</li>
              <li><strong>Read the uncertainty.</strong> Inspect the median and quantiles where available, or the probability of crossing a specified threshold.</li>
              <li><strong>Check past performance.</strong> Open <Link href="/metrics">Metrics</Link> for that product and compare results at the lead time you care about.</li>
            </ol>
          </section>

          <section id="forecasts" className={styles.section}>
            <h2>Reading forecasts</h2>
            <p>A forecast describes possible future conditions. An observation describes recorded conditions. Keep the two separate when comparing charts.</p>
            <div className={styles.cards}>
              <article className={styles.card}>
                <h3>Median & quantiles</h3>
                <p>The <strong>Median</strong> is the 50th percentile (q50), the model’s central estimate. <strong>Low</strong> and <strong>High</strong> are the 10th and 90th percentiles (q10 and q90), available in the chart tooltip.</p>
                <p>The q10–q90 range represents the model’s central 80% prediction interval. It is not a guaranteed minimum or maximum; actual coverage depends on model calibration.</p>
              </article>
              <article className={styles.card}>
                <h3>Threshold probabilities</h3>
                <p>Each heatmap row names an event, such as <strong>V ≥ 500 km/s</strong>. Each cell gives the model’s estimated probability for that event at a forecast lead time.</p>
                <p>A value of <strong>70%</strong> means a 70% estimated chance of meeting that threshold. It does not mean a 70% increase in speed or a 70% chance of infrastructure damage.</p>
              </article>
            </div>
            <dl className={styles.definitions}>
              <div><dt>Issue time</dt><dd>The reference time from which a forecast is made.</dd></div>
              <div><dt>Lead time</dt><dd>How far ahead a prediction looks, in hours after issue time. Lead hour 24 refers to one day ahead.</dd></div>
              <div><dt>Valid time</dt><dd>The time a prediction applies to. The API exposes this as <code>valid_time</code> for each prediction.</dd></div>
              <div><dt>Time zones</dt><dd>Live observations are labelled in UTC. Quantile chart dates and hours use your browser’s local time zone. Check the timestamp and time zone when comparing data.</dd></div>
            </dl>
          </section>

          <section id="products" className={styles.section}>
            <h2>Forecast products</h2>
            <p>Products differ in the quantities and forecast types they provide. Some variables may not yet have a forecast available.</p>
            <div className={styles.cards}>
              {products.map(product => (
                <article key={product.slug} className={styles.card}>
                  <span className={styles.badge}>{product.visibility === "public" ? "Public API" : "Private API"}</span>
                  <h3><Link href={`/products/${product.slug}`}>{product.title}</Link></h3>
                  <p>{product.description}</p>
                  <p className={styles.units}>{product.variables.map(variable => `${variable.label}: ${variable.unit}`).join(" · ")}</p>
                </article>
              ))}
            </div>
          </section>

          <section id="observations" className={styles.section}>
            <h2>Live data & freshness</h2>
            <p>The <Link href="/live">Live page</Link> shows the latest available hourly observations and up to 24 hourly records. It checks for updates every minute; this does not mean new measurements arrive every minute.</p>
            <dl className={styles.definitions}>
              <div><dt>Observed vs. last checked</dt><dd>“Observed” identifies the data timestamp. “Last checked” is when the page last successfully fetched data. A recent check can still return an older observation.</dd></div>
              <div><dt>Delayed data</dt><dd>A warning appears when the latest observation is more than three hours old. If a refresh fails, the page keeps the last loaded data and retries automatically.</dd></div>
              <div><dt>Missing values</dt><dd>A dash means a value is unavailable, not zero. Observations are normalized; missing measurements may be interpolated or filled during processing.</dd></div>
              <div><dt>S10, M10 & Y10</dt><dd>These are provisional daily solar index estimates calibrated from GOES data, updated using the UTC day’s available observations. Midnight closes the previous day.</dd></div>
            </dl>
          </section>

          <section id="metrics" className={styles.section}>
            <h2>Understanding model performance</h2>
            <p>The <Link href="/metrics">Metrics pages</Link> show evaluation results by forecast lead hour. Compare the same variable, threshold, and lead time: a single score does not describe every forecast situation.</p>
            <dl className={styles.definitions}>
              <div><dt>Brier score</dt><dd>Measures the error of event probabilities against observed outcomes. Lower is better; zero is perfect.</dd></div>
              <div><dt>ROC AUC</dt><dd>Measures how well the model ranks events above non-events. Higher is better; 0.5 corresponds to random ranking and 1 to perfect ranking. It does not measure probability calibration.</dd></div>
              <div><dt>Precision chart</dt><dd>This chart uses average precision, which summarizes the precision–recall trade-off across thresholds. Higher is better; interpret it alongside how common the event is.</dd></div>
              <div><dt>Reliability</dt><dd>Compares predicted probabilities with observed event frequencies. For a well-calibrated model, events assigned about 70% probability should occur about 70% of the time across many comparable predictions.</dd></div>
              <div><dt>Continuous metrics</dt><dd>For quantile forecasts, the available metrics are shown separately for each variable. MAE and RMSE, when provided, measure prediction error in the variable’s units; lower values are better.</dd></div>
            </dl>
            <p>Historical evaluation describes performance on past observations. It does not guarantee the accuracy of an individual forecast.</p>
          </section>

          <section id="api" className={styles.section}>
            <h2>API & support</h2>
            <p>Use the <a href="/api/v1/docs">API documentation</a> to explore endpoints and response schemas. Product pages link directly to their forecast and metrics JSON responses.</p>
            <div className={styles.resources}>
              <a href="/api/v1/public/observations/latest">Latest observations <span aria-hidden="true">↗</span></a>
              <a href="/api/v1/public/observations/history?limit=24">Observation history <span aria-hidden="true">↗</span></a>
              <a href="/api/v1/public/forecasts/solar-wind-speed">Solar wind forecast <span aria-hidden="true">↗</span></a>
              <a href="/api/v1/public/forecasts/solar-wind-speed/metrics">Solar wind metrics <span aria-hidden="true">↗</span></a>
            </div>
            <p>API probabilities are numbers from 0 to 1; heatmaps display them as percentages. Forecast responses include the available variables and prediction times, so check these before using a value.</p>
            <div className={styles.contact}>
              <h3>Questions or unexpected results?</h3>
              <p>Email <a href="mailto:krivenko.a.b@gmail.com">krivenko.a.b@gmail.com</a> or visit the <a href="https://github.com/a-kryvenko/argus" target="_blank" rel="noopener noreferrer">GitHub repository</a>. Include the page or endpoint, the relevant timestamp and time zone, and what you expected to see.</p>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
