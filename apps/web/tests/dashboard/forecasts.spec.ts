import { captureScreenshot } from "./screenshot";
import { test, expect, type Page } from '@playwright/test';
import { products, type ProductConfig } from '../../src/forecast/products';
import type { Forecast, ForecastPoint } from '../../src/forecast/types';

const issue = '2026-09-30T00:00:00.000Z';

test('old product URLs redirect to forecast and preserve query parameters', async ({ request }) => {
  for (const [source, destination] of [
    ['/products', '/forecast'],
    ['/products/solar-wind-speed', '/forecast/solar-wind-speed'],
    ['/demo/products', '/demo/forecast'],
    ['/demo/products/solar-events', '/demo/forecast/solar-events'],
  ]) {
    const response = await request.get(`${source}?offset=-24`, { maxRedirects: 0 });
    expect(response.status()).toBe(308);
    const location = new URL(response.headers().location, 'http://localhost:3100');
    expect(location.pathname).toBe(destination);
    expect(location.search).toBe('?offset=-24');
  }
});
const validTime = (lead: number) => new Date(Date.parse(issue) + lead * 3600000).toISOString();
function release(product: ProductConfig): Forecast {
  const horizon = product.apiTarget.startsWith('solar-wind') ? 96 : 48;
  const base: Record<string, number> = { v: 420, n: 6.25, bt: 7, bs: 3, kp: 3, ap: 14, dst: -20, f10_7: 155, s10: 95, m10: 100, y10: 110 };
  return { target: product.apiTarget, issue_time: issue, horizon_hours: horizon,
    available_variables: product.variables.map(variable => variable.key),
    predictions: Array.from({ length: horizon }, (_, i) => {
      const variables: ForecastPoint['variables'] = {};
      for (const variable of product.variables) {
        if (variable.key === 'bt' && i >= 24 || variable.key === 'n' && i === 1) continue;
        const amplitude = variable.key === 'v' ? 22 : variable.key === 'dst' ? 8 : 1;
        const median = base[variable.key] + (Math.sin(i / 9) + i / 96) * amplitude;
        variables[variable.key] = {
          continuous: variable.quantile ? { q10: median - amplitude * 1.5, q50: median, q90: median + amplitude * 1.5 } : null,
          binary: variable.thresholds.filter(threshold => !(variable.key === 'v' && threshold.value === 600 && i === 0)).map((threshold, j) => ({
            threshold: threshold.value,
            probability: variable.key === 'v' && threshold.value === 500 && i === 0 ? 0 : Math.max(0, Math.min(1, .4 - j * .15 + Math.sin(i / 13) * .22)),
          })),
        };
      }
      return { valid_time: validTime(i + 1), lead_hours: i + 1, variables };
    }),
  };
}
async function mockForecasts(page: Page, state: { failed?: boolean } = {}) {
  await page.route('**/api/v1/**/forecasts/**', async route => {
    const path = new URL(route.request().url()).pathname;
    const product = products.find(item => path.endsWith(`/forecasts/${item.apiTarget}`));
    if (!product) { await route.fulfill({ status: 404, json: {} }); return; }
    expect(path).toBe(`/api/v1/${product.visibility}/forecasts/${product.apiTarget}`);
    await route.fulfill(state.failed ? { status: 503, json: { success: false, data: null, error: { message: 'Not ready' } } }
      : { json: { success: true, data: release(product), error: null } });
  });
}
async function screenshot(page: Page, name: string) {
  await captureScreenshot(page, { path: `test-results/${name}.png`, fullPage: true });
}

test('forecast overview links summary, horizon, quantile interval and time inspector', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await mockForecasts(page);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Forecast overview', exact: true })).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Overview', exact: true })).toHaveAttribute('aria-current', 'page');
  const summary = page.getByRole('region', { name: 'Forecast summary' });
  await expect(summary).toContainText('420 km/s');
  await expect(summary).toContainText('6.25 cm⁻³');
  const inspector = page.getByRole('complementary', { name: 'Forecast inspector' });
  await expect(inspector).toContainText('0%');
  await expect(inspector).toContainText('—');
  await expect(page.locator('.recharts-area-area')).toBeVisible();
  await expect(page.getByRole('region', { name: 'Solar Wind Speed threshold probability', exact: true }).locator('canvas').first()).toBeVisible();
  await screenshot(page, 'forecast-overview-desktop');
  await page.getByRole('button', { name: 'First 24 hours', exact: true }).click();
  await expect(inspector.getByLabel('Forecast time · UTC')).toHaveAttribute('max', '23');
  await inspector.getByLabel('Forecast time · UTC').fill('11');
  await expect(inspector).toContainText('Lead +12 hours');
  await page.locator('summary').filter({ hasText: 'Hourly forecast values' }).click();
  await page.getByRole('button', { name: 'Inspect forecast at 30 Sept 2026, 02:00 UTC' }).click();
  await expect(inspector.getByLabel('Forecast time · UTC')).toHaveValue('1');
  await summary.getByRole('button', { name: 'Show Geomagnetic activity forecast', exact: true }).click();
  await expect(page.getByRole('group', { name: 'Forecast variable' }).getByRole('button', { name: 'AP Ap Index' })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Kp Index threshold probability', exact: true }).locator('canvas').first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('probability charts survive repeated variable and horizon changes', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.text().includes('has been disposed')) errors.push(message.text());
  });
  await mockForecasts(page);
  await page.goto('/forecast/geomagnetic-activity');
  const heatmap = page.getByRole('region', { name: 'Kp Index threshold probability', exact: true });
  await expect(heatmap.locator('canvas').first()).toBeVisible();
  for (let i = 0; i < 3; i++) {
    await page.getByRole('button', { name: 'AP Ap Index' }).click();
    await page.getByRole('button', { name: 'First 24 hours' }).click();
    await page.getByRole('button', { name: 'KP Kp Index' }).click();
    await page.getByRole('button', { name: 'Full forecast' }).click();
    await expect(heatmap.locator('canvas').first()).toBeVisible();
  }
  expect(errors).toEqual([]);
});

test('catalog lists all product definitions and uses the forecast workspace shell', async ({ page }) => {
  await page.goto('/forecast');
  await expect(page.getByRole('heading', { name: 'Forecast', exact: true })).toBeVisible();
  for (const product of products) await expect(page.getByRole('heading', { name: product.title, exact: true })).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Forecast', exact: true })).toHaveAttribute('aria-current', 'page');
  await screenshot(page, 'forecast-catalog-desktop');
});

test('forecast range supports keyboard selection and chart clicks do not take focus', async ({ page }) => {
  await mockForecasts(page);
  await page.goto('/forecast/solar-wind-speed');
  const slider = page.getByRole('slider', { name: 'Forecast time · UTC', exact: true });
  await expect(slider).toHaveAttribute('max', '95');
  await slider.press('End');
  await expect(slider).toHaveValue('95');
  await expect(slider).toHaveAttribute('aria-valuetext', /\+96 hours$/);
  await slider.press('Home');
  await slider.press('ArrowRight');
  await expect(slider).toHaveValue('1');
  await expect(slider).toHaveAttribute('aria-valuetext', /02:00 UTC · \+2 hours$/);
  const plot = page.getByRole('region', { name: 'Solar Wind Speed quantile forecast', exact: true });
  const chart = plot.locator('.recharts-surface');
  await chart.hover({ position: { x: 200, y: 100 } });
  await expect(plot.locator('.recharts-tooltip-wrapper')).toBeVisible();
  await chart.click({ position: { x: 200, y: 100 } });
  await expect(slider).toBeFocused();
  await expect(slider).not.toHaveValue('1');
  const heatmap = page.getByRole('region', { name: 'Solar Wind Speed threshold probability', exact: true }).locator('[data-chart]');
  await heatmap.click({ position: { x: 200, y: 30 } });
  await expect(slider).toBeFocused();
  await chart.focus();
  await expect(chart).toBeFocused();
});

test('forecast range follows available times with gaps and disables a single or empty timeline', async ({ page }) => {
  const forecast = release(products[0]);
  forecast.predictions = forecast.predictions.filter(point => [1, 5, 12].includes(point.lead_hours));
  await page.route('**/api/v1/public/forecasts/solar-wind-speed', route => route.fulfill({ json: { success: true, data: forecast, error: null } }));
  await page.goto('/forecast/solar-wind-speed');
  const slider = page.getByRole('slider', { name: 'Forecast time · UTC', exact: true });
  await expect(slider).toHaveAttribute('max', '2');
  await slider.fill('1');
  await expect(slider).toHaveAttribute('aria-valuetext', /05:00 UTC · \+5 hours$/);
  await slider.press('End');
  await expect(slider).toHaveAttribute('aria-valuetext', /12:00 UTC · \+12 hours$/);
  forecast.predictions = forecast.predictions.slice(0, 1);
  await page.reload();
  await expect(slider).toBeDisabled();
  await expect(slider).toHaveValue('0');
  await expect(slider).toHaveAttribute('aria-valuetext', /\+1 hours$/);
  forecast.predictions = [];
  await page.reload();
  await expect(slider).toBeDisabled();
  await expect(slider).toHaveAttribute('aria-valuetext', 'No forecast times');
});

test('density preserves unavailable quantiles instead of substituting zero', async ({ page }) => {
  await mockForecasts(page);
  await page.goto('/forecast/solar-wind-density');
  const inspector = page.getByRole('complementary', { name: 'Forecast inspector' });
  await expect(inspector).toContainText('6.25 cm⁻³');
  await inspector.getByLabel('Forecast time · UTC').fill('1');
  await expect(inspector).toContainText('No values for Plasma Density at this forecast time');
  await expect(inspector).toContainText('— cm⁻³');
  await page.locator('summary').filter({ hasText: 'Hourly forecast values' }).click();
  const row = page.getByRole('row').filter({ hasText: '2026-09-30 02:00' });
  await expect(row.getByRole('cell', { name: '—', exact: true })).toHaveCount(3);
});

test('HMF retains different variable horizons and resets selected time when filtered out', async ({ page }) => {
  await mockForecasts(page);
  await page.goto('/forecast/hmf');
  const inspector = page.getByRole('complementary', { name: 'Forecast inspector' });
  await inspector.getByLabel('Forecast time · UTC').fill('29');
  await expect(inspector).toContainText('No values for Total HMF at this forecast time');
  await page.getByRole('group', { name: 'Forecast variable' }).getByRole('button', { name: 'BS Southward Bz' }).click();
  await expect(inspector).not.toContainText('No values for');
  await expect(inspector).toContainText('Lead +30 hours');
  await page.getByRole('button', { name: 'First 24 hours', exact: true }).click();
  await expect(inspector.getByLabel('Forecast time · UTC')).toHaveValue('0');
  await expect(inspector.getByLabel('Forecast time · UTC')).toHaveAttribute('max', '23');
});

test('solar radiation switches all four quantile variables and retains units', async ({ page }) => {
  await mockForecasts(page);
  await page.goto('/forecast/solar-radiation');
  const inspector = page.getByRole('complementary', { name: 'Forecast inspector' });
  await expect(inspector).toContainText('155 sfu');
  await page.getByRole('group', { name: 'Forecast variable' }).getByRole('button', { name: 'M10 M10', exact: true }).click();
  await expect(inspector).toContainText('100 index');
  await expect(page.getByRole('region', { name: 'M10 quantile forecast', exact: true })).toBeVisible();
  await screenshot(page, 'forecast-product-desktop');
});

test('geomagnetic product switches from Kp probabilities to Ap quantiles', async ({ page }) => {
  await mockForecasts(page);
  await page.goto('/forecast/geomagnetic-activity');
  await expect(page.getByRole('region', { name: 'Kp Index threshold probability', exact: true })).toBeVisible();
  await page.getByRole('group', { name: 'Forecast variable' }).getByRole('button', { name: 'AP Ap Index' }).click();
  await expect(page.getByRole('region', { name: 'Ap Index quantile forecast', exact: true })).toBeVisible();
  await expect(page.getByRole('complementary', { name: 'Forecast inspector' })).toContainText('14 index');
});

test('Dst shows signed values and the actual release timestamp', async ({ page }) => {
  await mockForecasts(page);
  await page.goto('/forecast/dst');
  const inspector = page.getByRole('complementary', { name: 'Forecast inspector' });
  await expect(inspector).toContainText('-20 nT');
  await expect(inspector).toContainText('30 Sept 2026, 00:00 UTC');
  await expect(page.locator('.recharts-line-curve')).toBeVisible();
});

test('failed release read stays explicit and retries without inventing values', async ({ page }) => {
  const state = { failed: true };
  await mockForecasts(page, state);
  await page.goto('/forecast/solar-wind-speed');
  await expect(page.getByRole('alert').filter({ hasText: 'Could not load data' })).toContainText('Could not load data');
  await expect(page.getByRole('complementary', { name: 'Forecast inspector' })).toContainText('— km/s');
  state.failed = false;
  await page.getByRole('button', { name: 'Try again', exact: true }).click();
  await expect(page.locator('.recharts-line-curve')).toBeVisible();
  await expect(page.getByRole('alert').filter({ hasText: 'Could not load data' })).toHaveCount(0);
});

test('mobile forecast navigation, time inspector and charts fit the screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await mockForecasts(page);
  await page.goto('/forecast/solar-wind-speed');
  const inspector = page.getByRole('complementary', { name: 'Forecast inspector' });
  await expect(inspector.getByLabel('Forecast time · UTC')).toBeEnabled();
  await expect(inspector.getByLabel('Forecast time · UTC')).toBeHidden();
  await inspector.getByRole('button', { name: 'Details', exact: true }).click();
  await expect(inspector.getByLabel('Forecast time · UTC')).toBeVisible();
  await inspector.getByLabel('Forecast time · UTC').fill('5');
  await expect(inspector).toContainText('Lead +6 hours');
  await inspector.getByRole('button', { name: 'Details', exact: true }).click();
  await page.getByText('Resources', { exact: true }).click();
  await expect(page.getByRole('navigation', { name: 'Mobile resources' }).getByRole('link', { name: 'Help & methodology' })).toBeVisible();
  await page.getByText('Resources', { exact: true }).click();
  await expect(page.locator('.recharts-line-curve')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await screenshot(page, 'forecast-product-mobile');
});
