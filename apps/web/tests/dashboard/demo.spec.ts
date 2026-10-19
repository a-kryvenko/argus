import { test, expect } from '@playwright/test';

test('demo replays forecasts and preserves time between products', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', request => { if (request.url().includes('/api/v1/')) requests.push(request.url()); });
  await page.goto('/demo?offset=-48');
  await expect(page.getByRole('region', { name: 'Demo time controls' })).toBeVisible();
  await expect(page.getByText('T−48 h · Until event', { exact: true })).toBeVisible();
  await expect(page.getByText('17 Jan 2026, 19:38 UTC · Simulated now')).toBeVisible();
  await expect(page.getByRole('checkbox', { name: 'Show actual future values' })).toBeChecked();
  await page.getByRole('slider', { name: 'Hours relative to event' }).fill('-24');
  await expect(page).toHaveURL(/offset=-24$/);
  await expect(page.getByText('T−24 h · Until event', { exact: true })).toBeVisible();
  await page.getByRole('navigation', { name: 'Demo forecast products' }).getByRole('link', { name: 'Solar Wind Plasma Density' }).click();
  await expect(page).toHaveURL('/demo/forecast/solar-wind-density?offset=-24');
  await expect(page.getByRole('heading', { name: 'Solar Wind Plasma Density', exact: true }).first()).toBeVisible();
  await page.getByRole('checkbox', { name: 'Show actual future values' }).uncheck();
  await expect(page.getByText('Hidden cm⁻³')).toBeVisible();
  await page.getByRole('button', { name: 'T0', exact: true }).click();
  await expect(page.getByText('T0 · Event begins', { exact: true })).toBeVisible();
  await expect(page.getByText('19 Jan 2026, 19:38 UTC · Simulated now')).toBeVisible();
  expect(requests).toEqual([]);
});

test('demo observations preserve the replay clock without operational requests', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', request => { if (request.url().includes('/api/v1/')) requests.push(request.url()); });
  await page.goto('/demo/forecast/solar-wind-density?offset=0');
  await page.getByRole('link', { name: 'Observations', exact: true }).click();
  await expect(page).toHaveURL('/demo/live?offset=0');
  await expect(page.getByRole('heading', { name: 'Historical observations' })).toBeVisible();
  expect(requests).toEqual([]);
});

test('demo exits to the matching product', async ({ page }) => {
  await page.goto('/demo/forecast/solar-wind-density?offset=-24');
  await expect(page.getByRole('link', { name: 'Exit demo' })).toHaveAttribute('href', '/forecast/solar-wind-density');
  await page.route('**/api/v1/forecasts/solar-wind-density', route => route.fulfill({
    status: 503, json: { success: false, data: null, error: { message: 'No test forecast' } },
  }));
  await page.getByRole('link', { name: 'Exit demo' }).click();
  await expect(page).toHaveURL('/forecast/solar-wind-density');
  await expect(page.getByRole('region', { name: 'Demo time controls' })).toHaveCount(0);
});

test('panel remains visible without covering content on mobile and keyboard changes time', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/demo?offset=-96');
  const panel = page.getByRole('region', { name: 'Demo time controls' });
  const slider = page.getByRole('slider', { name: 'Hours relative to event' });
  await expect(slider).toBeEnabled();
  await slider.focus();
  await page.keyboard.press('ArrowRight');
  await expect(slider).toHaveValue('-95');
  await expect(page).toHaveURL(/offset=-95$/);
  await expect(page.getByText('T−95 h · Until event', { exact: true })).toBeVisible();
  const panelBounds = await panel.boundingBox();
  const navBounds = await page.getByRole('complementary', { name: 'Workspace navigation' }).boundingBox();
  expect(navBounds!.y).toBeGreaterThanOrEqual(panelBounds!.height);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.evaluate(() => window.scrollTo(0, 800));
  expect((await panel.boundingBox())!.y).toBe(0);
  await expect(page.getByRole('link', { name: 'Exit demo' })).toBeVisible();
});

test('invalid offsets recover safely, unavailable products do not show a different forecast', async ({ page }) => {
  await page.goto('/demo/forecast/dst?offset=oops');
  await expect(page.getByRole('slider', { name: 'Hours relative to event' })).toHaveValue('-96');
  await expect(page.getByRole('heading', { name: 'Dst Index', exact: true })).toBeVisible();
  await expect(page.getByText('No demo release is available at this time.')).toBeVisible();
  await page.goto('/demo/not-a-page');
  await expect(page.getByRole('region', { name: 'Demo time controls' })).toHaveCount(0);
});

test('playback stops at T+72 and can restart from the beginning', async ({ page }) => {
  await page.goto('/demo?offset=71');
  await page.getByRole('button', { name: 'Play replay' }).click();
  await expect(page.getByRole('slider', { name: 'Hours relative to event' })).toHaveValue('72');
  await expect(page.getByRole('button', { name: 'Play replay' })).toBeVisible();
  await expect(page.getByText('T+72 h · Since event', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Play replay' }).click();
  await expect(page.getByRole('slider', { name: 'Hours relative to event' })).toHaveValue('-96');
  await page.getByRole('button', { name: 'Pause replay' }).click();
  await expect(page.getByRole('button', { name: 'Play replay' })).toBeVisible();
});

test('demo catalog links geomagnetic forecasts and replays probabilities', async ({ page }) => {
  const operationalRequests: string[] = [];
  page.on('request', request => { if (request.url().includes('/api/v1/')) operationalRequests.push(request.url()); });
  await page.goto('/demo/forecast?offset=-48');
  await expect(page.getByRole('link', { name: /Geomagnetic Activity.*Explore demo forecast/ })).toBeVisible();
  await page.goto('/demo/forecast/geomagnetic-activity?offset=-48');
  await expect(page.getByRole('region', { name: 'Kp Index forecast and observations' })).toBeVisible();
  await expect(page.getByText('Median · q50', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'AP Ap Index' }).click();
  await expect(page.getByRole('region', { name: 'Ap Index forecast and observations' })).toBeVisible();
  expect(operationalRequests).toEqual([]);
});

for (const product of [
  { slug: 'dst', chart: 'Dst Index', value: '-45' },
  { slug: 'hmf', chart: 'Total HMF', tab: 'BS Southward Bz', missing: 'Southward Bz' },
  { slug: 'solar-radiation', chart: 'F10.7', tab: 'S10 S10', missing: 'S10' },
]) test(`demo ${product.slug} preserves signed values and partial model availability`, async ({ page }) => {
  const requests: string[] = [];
  page.on('request', request => { if (request.url().includes('/api/v1/')) requests.push(request.url()); });
  await page.goto(`/demo/forecast/${product.slug}?offset=-48`);
  await expect(page.getByRole('region', { name: `${product.chart} forecast and observations` })).toBeVisible();
  if (product.value) await expect(page.getByText(product.value, { exact: false }).first()).toBeVisible();
  if (product.tab) {
    await page.getByRole('button', { name: product.tab }).click();
    await expect(page.getByText(`${product.missing} is not available in this release.`)).toBeVisible();
  }
  expect(requests).toEqual([]);
});

test('demo controls wait for hydration before accepting replay input', async ({ page }) => {
  let releaseScripts!: () => void;
  const scriptsReady = new Promise<void>(resolve => { releaseScripts = resolve; });
  await page.route('**/_next/**/*.js*', async route => {
    await scriptsReady;
    await route.continue();
  });
  await page.goto('/demo?offset=-96', { waitUntil: 'commit' });
  const slider = page.getByRole('slider', { name: 'Hours relative to event' });
  try {
    await expect(slider).toBeVisible();
    await expect(slider).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Play replay' })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'T0', exact: true })).toBeDisabled();
    await expect(page.getByRole('checkbox', { name: 'Show actual future values' })).toBeDisabled();
  } finally {
    releaseScripts();
  }
  await expect(slider).toBeEnabled();
  await slider.press('ArrowRight');
  await expect(slider).toHaveValue('-95');
  await expect(page).toHaveURL(/offset=-95$/);
  await expect(page.getByText('T−95 h · Until event', { exact: true })).toBeVisible();
});

test('LEO scenario switches chains, hides future outcomes and preserves the replay clock', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', request => { if (request.url().includes('/api/v1/')) requests.push(request.url()); });
  await page.goto('/demo?offset=-48');
  await page.getByRole('link', { name: 'Explore LEO density scenario →' }).click();
  await expect(page).toHaveURL('/demo/leo?offset=-48');
  await expect(page.getByRole('heading', { name: 'LEO density scenario' })).toBeVisible();
  await expect(page.getByRole('status', { name: 'Density attention' })).toHaveText('Below review threshold');
  await page.getByLabel('Geomagnetic drivers').selectOption('forecast');
  await expect(page.getByRole('status', { name: 'Density attention' })).toHaveText('Review');
  await page.getByLabel('Atmosphere model').selectOption('dtm2020');
  await expect(page.getByRole('status', { name: 'Density attention' })).toHaveText('High');
  await expect(page.getByRole('region', { name: 'Atmospheric density forecast and observations' })).toBeVisible();
  await expect(page.getByText('Full replay evaluation · Includes future outcomes and later issues')).toBeVisible();
  await page.getByRole('checkbox', { name: 'Show actual future values' }).uncheck();
  await expect(page.getByText('Full replay evaluation · Includes future outcomes and later issues')).toHaveCount(0);
  await expect(page.getByText(/Gold: actual future \(hidden\)/)).toBeVisible();
  await expect(page.getByRole('link', { name: 'Exit demo' })).toHaveAttribute('href', '/dashboard/risk/leo');
  expect(requests).toEqual([]);
});

test('LEO missing issue remains unavailable and mobile content does not overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/demo/leo?offset=-95');
  await page.getByLabel('Atmosphere model').selectOption('dtm2020');
  await page.getByLabel('Geomagnetic drivers').selectOption('forecast');
  await expect(page.getByRole('status', { name: 'Density attention' })).toHaveText('No release for this hour');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test('historical ENLIL arrivals appear only after publication and revisions follow the clock', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', request => { if (request.url().includes('/api/v1/')) requests.push(request.url()); });
  await page.goto('/demo/forecast/solar-wind-speed?offset=-22');
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Solar events · CME' }).click();
  await expect(page).toHaveURL('/demo/forecast/solar-events?offset=-22');
  await expect(page.getByText('CME synthetic', { exact: true })).toHaveCount(0);
  await page.getByRole('slider', { name: 'Hours relative to event' }).fill('-21');
  await expect(page.getByText('CME synthetic', { exact: true })).toBeVisible();
  await expect(page.getByText('20 Jan 2026, 02:18 UTC', { exact: true })).toBeVisible();
  await page.getByRole('slider', { name: 'Hours relative to event' }).fill('-11');
  await expect(page.getByText('20 Jan 2026, 01:07 UTC', { exact: true })).toBeVisible();
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'LEO density scenario' }).click();
  await expect(page.getByRole('region', { name: 'Historical CME arrival context' })).toBeVisible();
  expect(requests).toEqual([]);
});
