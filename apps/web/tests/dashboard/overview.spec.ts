import { test, expect, type Page } from '@playwright/test';
import { captureScreenshot } from './screenshot';
import { mockSDO, sdoFixture } from '../fixtures/sdo';

const time = '2026-10-30T12:00:00Z';
const image = `${'a'.repeat(64)}.svg`;
function fixture() {
  const item = (title: string, value: number | null, risk_level: number, measurements = '', image_name: string | null = null) =>
    ({ title, value, risk_level, measurements, image_name, observed_at: value === null ? null : time });
  return {
    generated_at: time, cme_events_available: true, sdo_images: sdoFixture(),
    cme_events: [{ event_id: 'CME-2026-10-29', source_url: 'https://example.org/cme', observed_at: time,
      arrival_window_start: time, arrival_window_end: null }],
    observations: {
      plasma: { title: 'Plasma', items: [item('v', 540, 2, 'km/s', image), item('n', 8.25, 0, 'cm⁻³'), item('t', null, 0, 'K')] },
      geo: { title: 'Geomagnetic', items: [item('Kp', 5, 1), item('Ap', 48, 1), item('Dst', -160, 4, 'nT')] },
      hmf: { title: 'HMF', items: [item('Bt', 16, 3, 'nT'), item('Bz', -8, 1, 'nT')] },
      radio: { title: 'Radio', items: [item('F10.7', 155.26, 0, 'sfu')] },
    },
  };
}
async function mock(page: Page, data = fixture()) {
  await mockSDO(page);
  await page.route('**/api/v1/summary', route => route.fulfill({ json: { success: true, data, error: null } }));
  await page.route('**/api/v1/summary/images/*', route => route.fulfill({ contentType: 'image/svg+xml',
    body: '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="240"><path d="M0 200 L240 120 L480 40" stroke="orange" fill="none"/></svg>' }));
}

test('homepage shows CME, ordered groups, risks and API SVGs without forecast requests', async ({ page }) => {
  const errors: string[] = [];
  const forecastRequests: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (request.url().includes('/api/v1/forecasts/')) forecastRequests.push(request.url()); });
  await mock(page);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Space weather' })).toBeVisible();
  await expect(page.locator('main h2')).toHaveText(['CME events', 'Plasma', 'Geomagnetic', 'HMF', 'Radio']);
  await expect(page.locator('main > :first-child h1')).toHaveText('Space weather');
  await expect(page.locator('main > header + section')).toHaveAttribute('aria-label', 'SDO AIA');
  await expect(page.getByRole('link', { name: /CME-2026-10-29/ })).toHaveAttribute('href', 'https://example.org/cme');
  const plasma = page.getByRole('region', { name: 'Plasma', exact: true });
  await expect(plasma).toHaveAttribute('data-risk', '2');
  await expect(plasma).toContainText('540');
  await expect(plasma).toContainText('km/s');
  const chart = plasma.getByRole('img');
  await expect(chart).toHaveAttribute('src', `/api/v1/summary/images/${image}`);
  await expect.poll(() => chart.evaluate(node => (node as HTMLImageElement).naturalWidth)).toBe(480);
  await expect(page.getByRole('region', { name: 'Geomagnetic', exact: true })).toHaveAttribute('data-risk', '4');
  await expect(page.getByRole('region', { name: 'Radio', exact: true })).toHaveAttribute('data-risk', '0');
  await captureScreenshot(page, { path: 'test-results/overview-desktop.png', fullPage: true });
  expect(errors).toEqual([]);
  expect(forecastRequests).toEqual([]);
});

test('homepage grid shrinks and stacks without page overflow', async ({ page }) => {
  await mock(page);
  await page.goto('/');
  const plasma = page.getByRole('region', { name: 'Plasma', exact: true });
  const geo = page.getByRole('region', { name: 'Geomagnetic', exact: true });
  const hmf = page.getByRole('region', { name: 'HMF', exact: true });
  await expect(plasma).toBeVisible();
  const navigation = page.getByRole('navigation', { name: 'Main navigation' });
  await expect(navigation.locator('svg')).toHaveCount(0);
  for (const width of [1440, 900, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await navigation.getByRole('link', { name: 'Overview', exact: true }).locator('span').first()
      .evaluate(node => getComputedStyle(node).display)).not.toBe('none');
    const a = (await plasma.boundingBox())!;
    const b = (await geo.boundingBox())!;
    const c = (await hmf.boundingBox())!;
    if (width > 760) { expect(a.y).toBe(b.y); expect(b.x).toBeGreaterThan(a.x); expect(c.x).toBe(a.x); }
    else { expect(a.x).toBe(b.x); expect(b.y).toBeGreaterThan(a.y); expect(c.y).toBeGreaterThan(b.y); }
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await captureScreenshot(page, { path: 'test-results/overview-mobile.png', fullPage: true });
});

test('homepage hides empty CME blocks and handles failed charts', async ({ page }) => {
  const data = fixture();
  data.cme_events = [];
  await mock(page, data);
  await page.route('**/api/v1/summary/images/*', route => route.fulfill({ status: 404 }));
  await page.goto('/');
  const plasma = page.getByRole('region', { name: 'Plasma', exact: true });
  await expect(plasma).toBeVisible();
  await expect(page.getByRole('region', { name: 'CME events', exact: true })).toHaveCount(0);
  await expect(plasma.getByRole('img')).toHaveCount(0);
  await expect(plasma).toContainText('540');
  data.cme_events_available = false;
  await page.reload();
  await expect(plasma).toBeVisible();
  await expect(page.getByRole('region', { name: 'CME events', exact: true })).toHaveCount(0);
});

test('homepage can retry an unavailable summary', async ({ page }) => {
  await page.route('**/api/v1/summary', route => route.fulfill({ status: 503, json: { success: false, data: null, error: null } }));
  await page.goto('/');
  await expect(page.getByRole('main').getByRole('alert')).toContainText('Could not load data');
  await mock(page);
  await page.getByRole('button', { name: 'Try again', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Plasma', exact: true })).toBeVisible();
});
