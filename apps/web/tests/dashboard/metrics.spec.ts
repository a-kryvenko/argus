import { captureScreenshot } from "./screenshot";
import { test, expect, type Page } from '@playwright/test';
import { products, type ProductConfig } from '../../app/_config/products';
import type { ForecastMetrics } from '../../app/_utils/api';

function evaluation(product: ProductConfig): ForecastMetrics {
  return { target: product.apiTarget, variables: Object.fromEntries(product.variables.map(variable => [variable.key, {
    continuous: variable.quantile ? { quantiles: [.1, .5, .9], by_lead_hour: Array.from({ length: 48 }, (_, i) => ({
      lead_hours: i + 1, values: { mae: i === 5 ? null : 20 + i / 2, rmse: 30 + i / 2, coverage_80: .72 + i / 1000, n: 1000 - i, bias: -2, ...(i ? { q50_pinball: 10 + i / 4 } : {}) },
    })) } : null,
    binary: variable.thresholds.map((threshold, j) => ({ threshold: threshold.value, operator: 'gte' as const,
      by_lead_hour: Array.from({ length: j === 2 ? 24 : 48 }, (_, i) => ({ lead_hours: i + 1,
        brier_score: i === 0 && j === 0 ? 0 : .12 + j * .04 + i / 1000,
        roc_auc: i === 5 ? null : .89 - i / 1000, average_precision: .7 - j * .08,
        threat_score: .6, heidke_skill_score: -.15,
        reliability: i === 5 ? [] : [.1, .3, .5, .7, .9].map(p => ({ predicted_probability: p, observed_frequency: Math.min(1, p + j * .02 + i / 1000) })),
      })).filter(row => !(j === 1 && row.lead_hours === 3)).reverse(),
    })),
  }])) };
}
async function mock(page: Page, state: { failed?: boolean; empty?: boolean; missing?: boolean } = {}) {
  await page.route('**/api/v1/**/forecasts/**/metrics', async route => {
    const path = new URL(route.request().url()).pathname;
    const product = products.find(item => path.endsWith(`/forecasts/${item.apiTarget}/metrics`))!;
    expect(path).toBe(`/api/v1/${product.visibility}/forecasts/${product.apiTarget}/metrics`);
    const data = evaluation(product);
    if (state.empty) data.variables = {};
    if (state.missing) delete data.variables[product.variables[0].key];
    await route.fulfill(state.failed ? { status: 503, json: { success: false, data: null, error: { message: 'Unavailable' } } } : { json: { success: true, data, error: null } });
  });
}
async function screenshot(page: Page, name: string) {
  await captureScreenshot(page, { path: `test-results/${name}.png`, fullPage: true });
}

test('metrics catalog uses the workspace and links all six products', async ({ page }) => {
  await page.goto('/metrics');
  await expect(page.getByRole('heading', { name: 'Model performance', exact: true })).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Model performance' })).toHaveAttribute('aria-current', 'page');
  for (const product of products) await expect(page.getByRole('link').filter({ has: page.getByRole('heading', { name: product.title, exact: true }) })).toHaveAttribute('href', `/metrics/${product.slug}`);
  await screenshot(page, 'metrics-catalog-desktop');
});

test('continuous and threshold views share selected lead, preserve zero and expose all metrics', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await mock(page); await page.goto('/metrics/solar-wind-speed');
  const inspector = page.getByRole('complementary', { name: 'Metrics inspector' });
  await expect(inspector).toContainText('1,000');
  await expect(inspector.getByText('0', { exact: true })).toBeVisible();
  await expect(inspector.getByRole('heading', { name: 'Brier score', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Brier score by lead hour', exact: true }).locator('.recharts-line-curve')).toHaveCount(3);
  await expect(page.getByLabel('Continuous metric').locator('option', { hasText: 'q50 pinball' })).toHaveCount(1);
  await screenshot(page, 'metrics-product-desktop');
  const chart = page.getByRole('region', { name: 'MAE by lead hour' });
  await chart.locator('.recharts-line-dot').last().click();
  await expect(inspector.getByLabel('Evaluation lead hour')).toHaveAttribute('aria-valuetext', '+48 hours');
  await expect(page.getByRole('region', { name: 'Reliability at selected lead hour' })).toContainText('Lead +48h');
  const slider = inspector.getByRole('slider', { name: 'Evaluation lead hour' });
  await slider.scrollIntoViewIfNeeded();
  const track = (await slider.boundingBox())!;
  await page.mouse.move(track.x + track.width - 8, track.y + track.height / 2);
  await page.mouse.down();
  await page.mouse.move(track.x + track.width / 2, track.y + track.height / 2, { steps: 8 });
  await page.mouse.up();
  const selectedLead = Number(await slider.inputValue()) + 1;
  expect(selectedLead).toBeGreaterThan(1);
  expect(selectedLead).toBeLessThan(48);
  await expect(page.getByRole('region', { name: 'Selected lead metrics' })).toContainText(`Lead +${selectedLead}h`);
  await inspector.getByLabel('Evaluation lead hour').focus();
  await page.keyboard.press('Home');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  await expect(inspector).toContainText('—');
  await expect(page.getByRole('region', { name: 'Reliability at selected lead hour' })).toContainText('Lead +3h');
  await page.getByLabel('Threshold metric').selectOption('heidke_skill_score');
  await expect(inspector).toContainText('-0.15');
  await expect(page.getByRole('region', { name: 'Heidke skill score by lead hour' })).toBeVisible();
  await page.getByLabel('Continuous metric').selectOption('bias');
  await expect(page.getByRole('region', { name: 'Bias by lead hour' })).toBeVisible();
  await page.getByText('Values by lead hour', { exact: false }).first().click();
  await page.getByRole('button', { name: 'Inspect lead 6 hours', exact: true }).click();
  await expect(inspector.getByLabel('Evaluation lead hour')).toHaveAttribute('aria-valuetext', '+6 hours');
  await expect(page.getByRole('region', { name: 'Reliability at selected lead hour' })).toContainText('No reliability points at this lead hour');
  await expect(inspector.getByRole('definition').filter({ hasText: /^—$/ })).toHaveCount(1);
  expect(errors).toEqual([]);
});

test('binary-only HMF uses real lead hours, horizon reset and missing threshold values', async ({ page }) => {
  await mock(page); await page.goto('/metrics/hmf');
  const inspector = page.getByRole('complementary', { name: 'Metrics inspector' });
  await inspector.getByLabel('Evaluation lead hour').fill('29');
  await expect(inspector).toContainText('—');
  await expect(page.getByRole('region', { name: 'Reliability at selected lead hour' })).toContainText('Lead +30h');
  await page.getByRole('button', { name: 'First 24 hours', exact: true }).click();
  await expect(inspector.getByLabel('Evaluation lead hour')).toHaveAttribute('aria-valuetext', '+1 hours');
  await expect(inspector.getByLabel('Evaluation lead hour')).toHaveAttribute('max', '23');
  await page.getByRole('button', { name: 'BS Southward Bz', exact: true }).click();
  await expect(inspector).toContainText('Southward Bz');
  await expect(page.getByLabel('Continuous metric')).toHaveCount(0);
});

test('missing variable can switch to available evaluation without inventing metrics', async ({ page }) => {
  await mock(page, { missing: true }); await page.goto('/metrics/geomagnetic-activity');
  await expect(page.getByRole('status').filter({ hasText: 'Kp Index metrics are not available' })).toBeVisible();
  await page.getByRole('button', { name: 'AP Ap Index', exact: true }).click();
  await expect(page.getByRole('region', { name: 'MAE by lead hour' })).toBeVisible();
  await expect(page.getByLabel('Threshold metric')).toHaveCount(0);
});

for (const slug of ['solar-radiation', 'solar-wind-density', 'dst']) test(`${slug} renders continuous metrics and correct forecast link`, async ({ page }) => {
  await mock(page); await page.goto(`/metrics/${slug}`);
  await expect(page.getByRole('region', { name: 'MAE by lead hour' }).locator('.recharts-line-curve')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Open forecast', exact: true }).first()).toHaveAttribute('href', `/products/${slug}`);
  if (slug === 'solar-radiation') {
    await page.getByRole('button', { name: 'Y10 Y10', exact: true }).click();
    await expect(page.getByRole('complementary', { name: 'Metrics inspector' })).toContainText('Y10');
  }
});

test('failed metrics retry and empty response remain explicit', async ({ page }) => {
  const state = { failed: true, empty: false }; await mock(page, state); await page.goto('/metrics/dst');
  await expect(page.getByRole('alert').filter({ hasText: 'Could not load data' })).toBeVisible();
  state.failed = false; state.empty = true;
  await page.getByRole('button', { name: 'Try again', exact: true }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Dst Index metrics are not available' })).toBeVisible();
  state.empty = false;
  await page.getByRole('button', { name: 'Refresh', exact: true }).click();
  await expect(page.getByRole('region', { name: 'MAE by lead hour' })).toBeVisible();
});

test('mobile inspector, metric controls and charts fit narrow screens', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await mock(page); await page.goto('/metrics/solar-wind-speed');
  const inspector = page.getByRole('complementary', { name: 'Metrics inspector' });
  await expect(inspector.getByLabel('Evaluation lead hour')).toBeHidden();
  await inspector.getByRole('button', { name: 'Details', exact: true }).click();
  await inspector.getByLabel('Evaluation lead hour').fill('11');
  await expect(page.getByRole('region', { name: 'Reliability at selected lead hour' })).toContainText('Lead +12h');
  await inspector.getByRole('button', { name: 'Details', exact: true }).click();
  await page.getByLabel('Threshold metric').selectOption('average_precision');
  await expect(page.getByRole('region', { name: 'Average precision by lead hour' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await screenshot(page, 'metrics-product-mobile');
});

test('monthly observed accuracy is independent of static metrics and preserves zero', async ({ page }) => {
  await mock(page, { failed: true });
  await page.route('**/api/v1/**/forecasts/**/verification', route => route.fulfill({ json: {
    success: true, error: null, data: {
      product: 'solar-wind-speed', start: '2026-09-03T00:00:00Z', end: '2026-10-03T00:00:00Z',
      groups: [
        { artifact: 'plasma_speed_quantile', releases: 48,
          evaluated_at: '2026-10-03T00:00:00Z', counts: { verified: 100, missing: 2, pending: 3 },
          by_lead_hour: [
            { lead_hours: 3, counts: { verified: 80, missing: 2, pending: 3 }, continuous: { mae: 12.5, rmse: 18.5 }, binary: {} },
            { lead_hours: 48, counts: { verified: 20, missing: 0, pending: 0 }, continuous: { mae: 42, rmse: 50 }, binary: {} },
          ] },
        { artifact: 'plasma_speed_threshold', releases: 48,
          evaluated_at: '2026-10-03T00:00:00Z', counts: { verified: 100, missing: 2, pending: 3 },
          by_lead_hour: [{ lead_hours: 3, counts: { verified: 100, missing: 2, pending: 3 }, continuous: null, binary: { '450': { brier: 0 }, '500': { brier: .000004949486147261768 } } }] },
      ],
    },
  } }));
  await page.goto('/metrics/solar-wind-speed');
  const monthly = page.getByRole('region', { name: 'Last 30 days average accuracy' });
  await expect(monthly.getByTitle('12.5', { exact: true })).toHaveText('13');
  await expect(monthly.getByTitle('18.5', { exact: true })).toHaveText('19');
  await expect(monthly.getByTitle('0.000004949486147261768', { exact: true })).toHaveText('<0.001');
  await expect(monthly).toContainText('100 verified pairs');
  await expect(monthly.getByRole('heading', { name: 'Solar Wind Speed · Quantile forecast', exact: true })).toHaveCount(1);
  await expect(monthly).toContainText('All model versions are combined for this period.');
  await expect(monthly).toContainText('2026-09-03 00:00 UTC');
  await expect(monthly.getByRole('table').last().getByText('0', { exact: true })).toBeVisible();
  await expect(monthly).toContainText('Brier score · ≥ 450 km/s');
  const continuousTable = monthly.getByRole('table').first();
  await expect(continuousTable.getByRole('row').filter({ hasText: '+3h' })).toContainText('13');
  await expect(continuousTable.getByRole('row').filter({ hasText: '+48h' })).toContainText('42');
  await expect(continuousTable.getByRole('row').filter({ hasText: '+96h' })).toContainText('—');
  await expect(continuousTable.getByRole('row')).toHaveCount(7);
  await expect(monthly.getByRole('columnheader', { name: 'Missing', exact: true })).toHaveCount(0);
  await expect(monthly.getByRole('columnheader', { name: 'Pending', exact: true })).toHaveCount(0);
  await page.route('**/api/v1/**/forecasts/**/verification', route => route.fulfill({ json: {
    success: true, data: { start: '2026-09-03T00:00:00Z', end: '2026-10-03T00:00:00Z', groups: [] }, error: null,
  } }));
  await page.reload();
  await expect(monthly).toContainText('No verified forecast data available for this period.');
});
