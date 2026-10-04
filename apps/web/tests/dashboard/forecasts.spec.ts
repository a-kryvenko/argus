import { captureScreenshot } from "./screenshot";
import { test, expect, type Page } from '@playwright/test';
import { products, type ProductConfig } from '../../app/_config/products';
import type { Forecast, ForecastPoint } from '../../app/_utils/api';

const issue = '2026-09-30T00:00:00.000Z';
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
  await expect(page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Forecast overview' })).toHaveAttribute('aria-current', 'page');
  const summary = page.getByRole('region', { name: 'Forecast products summary' });
  await expect(summary).toContainText('420 km/s');
  await expect(summary).toContainText('6.25 cm⁻³');
  const inspector = page.getByRole('complementary', { name: 'Forecast inspector' });
  await expect(inspector).toContainText('0%');
  await expect(inspector).toContainText('—');
  await expect(page.locator('.recharts-area-area')).toBeVisible();
  await expect(page.getByRole('region', { name: 'Solar Wind Speed threshold probability', exact: true }).locator('canvas').first()).toBeVisible();
  await screenshot(page, 'forecast-overview-desktop');
  await page.getByRole('button', { name: 'First 24 hours', exact: true }).click();
  await expect(inspector.getByLabel('Forecast time · UTC').locator('option')).toHaveCount(24);
  await inspector.getByLabel('Forecast time · UTC').selectOption(validTime(12));
  await expect(inspector).toContainText('Lead +12 hours');
  await page.locator('summary').filter({ hasText: 'Hourly forecast values' }).click();
  await page.getByRole('button', { name: 'Inspect forecast at 30 Sept 2026, 02:00 UTC' }).click();
  await expect(inspector.getByLabel('Forecast time · UTC')).toHaveValue(validTime(2));
  await summary.getByRole('button', { name: 'Show Geomagnetic activity forecast', exact: true }).click();
  await expect(page.getByRole('group', { name: 'Forecast variable' }).getByRole('button', { name: 'AP Ap Index' })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Kp Index threshold probability', exact: true }).locator('canvas').first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('catalog lists all product definitions and uses the forecast workspace shell', async ({ page }) => {
  await page.goto('/products');
  await expect(page.getByRole('heading', { name: 'Forecast products', exact: true })).toBeVisible();
  for (const product of products) await expect(page.getByRole('heading', { name: product.title, exact: true })).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Forecast products' })).toHaveAttribute('aria-current', 'page');
  await screenshot(page, 'forecast-catalog-desktop');
});

test('density preserves unavailable quantiles instead of substituting zero', async ({ page }) => {
  await mockForecasts(page);
  await page.goto('/products/solar-wind-density');
  const inspector = page.getByRole('complementary', { name: 'Forecast inspector' });
  await expect(inspector).toContainText('6.25 cm⁻³');
  await inspector.getByLabel('Forecast time · UTC').selectOption(validTime(2));
  await expect(inspector).toContainText('No values for Plasma Density at this forecast time');
  await expect(inspector).toContainText('— cm⁻³');
  await page.locator('summary').filter({ hasText: 'Hourly forecast values' }).click();
  const row = page.getByRole('row').filter({ hasText: '2026-09-30 02:00' });
  await expect(row.getByRole('cell', { name: '—', exact: true })).toHaveCount(3);
});

test('HMF retains different variable horizons and resets selected time when filtered out', async ({ page }) => {
  await mockForecasts(page);
  await page.goto('/products/hmf');
  const inspector = page.getByRole('complementary', { name: 'Forecast inspector' });
  await inspector.getByLabel('Forecast time · UTC').selectOption(validTime(30));
  await expect(inspector).toContainText('No values for Total HMF at this forecast time');
  await page.getByRole('group', { name: 'Forecast variable' }).getByRole('button', { name: 'BS Southward Bz' }).click();
  await expect(inspector).not.toContainText('No values for');
  await expect(inspector).toContainText('Lead +30 hours');
  await page.getByRole('button', { name: 'First 24 hours', exact: true }).click();
  await expect(inspector.getByLabel('Forecast time · UTC')).toHaveValue(validTime(1));
  await expect(inspector.getByLabel('Forecast time · UTC').locator('option')).toHaveCount(24);
});

test('solar radiation switches all four quantile variables and retains units', async ({ page }) => {
  await mockForecasts(page);
  await page.goto('/products/solar-radiation');
  const inspector = page.getByRole('complementary', { name: 'Forecast inspector' });
  await expect(inspector).toContainText('155 sfu');
  await page.getByRole('group', { name: 'Forecast variable' }).getByRole('button', { name: 'M10 M10', exact: true }).click();
  await expect(inspector).toContainText('100 index');
  await expect(page.getByRole('region', { name: 'M10 quantile forecast', exact: true })).toBeVisible();
  await screenshot(page, 'forecast-product-desktop');
});

test('geomagnetic product switches from Kp probabilities to Ap quantiles', async ({ page }) => {
  await mockForecasts(page);
  await page.goto('/products/geomagnetic-activity');
  await expect(page.getByRole('region', { name: 'Kp Index threshold probability', exact: true })).toBeVisible();
  await page.getByRole('group', { name: 'Forecast variable' }).getByRole('button', { name: 'AP Ap Index' }).click();
  await expect(page.getByRole('region', { name: 'Ap Index quantile forecast', exact: true })).toBeVisible();
  await expect(page.getByRole('complementary', { name: 'Forecast inspector' })).toContainText('14 index');
});

test('Dst shows signed values and the actual release timestamp', async ({ page }) => {
  await mockForecasts(page);
  await page.goto('/products/dst');
  const inspector = page.getByRole('complementary', { name: 'Forecast inspector' });
  await expect(inspector).toContainText('-20 nT');
  await expect(inspector).toContainText('30 Sept 2026, 00:00 UTC');
  await expect(page.locator('.recharts-line-curve')).toBeVisible();
});

test('failed release read stays explicit and retries without inventing values', async ({ page }) => {
  const state = { failed: true };
  await mockForecasts(page, state);
  await page.goto('/products/solar-wind-speed');
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
  await page.goto('/products/solar-wind-speed');
  const inspector = page.getByRole('complementary', { name: 'Forecast inspector' });
  await expect(inspector.getByLabel('Forecast time · UTC')).toBeEnabled();
  await expect(inspector.getByLabel('Forecast time · UTC')).toBeHidden();
  await inspector.getByRole('button', { name: 'Details', exact: true }).click();
  await expect(inspector.getByLabel('Forecast time · UTC')).toBeVisible();
  await inspector.getByLabel('Forecast time · UTC').selectOption(validTime(6));
  await expect(inspector).toContainText('Lead +6 hours');
  await inspector.getByRole('button', { name: 'Details', exact: true }).click();
  await page.getByText('Resources', { exact: true }).click();
  await expect(page.getByRole('navigation', { name: 'Mobile resources' }).getByRole('link', { name: 'Help & methodology' })).toBeVisible();
  await page.getByText('Resources', { exact: true }).click();
  await expect(page.locator('.recharts-line-curve')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await screenshot(page, 'forecast-product-mobile');
});
