import { test, expect } from '@playwright/test';

test('demo replays forecasts, preserves time between sections and exits to the matching product', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', request => { if (request.url().includes('/api/v1/')) requests.push(request.url()); });
  await page.goto('/demo?offset=-48');
  await expect(page.getByRole('region', { name: 'Demo time controls' })).toBeVisible();
  await expect(page.getByText('T−48 h · Until event', { exact: true })).toBeVisible();
  await expect(page.getByText('17 Jan 2026, 19:38 UTC · Simulated now')).toBeVisible();
  await expect(page.getByRole('checkbox', { name: 'Show actual future values' })).toBeChecked();
  await page.getByRole('slider', { name: 'Hours before event' }).fill('-24');
  await expect(page).toHaveURL(/offset=-24$/);
  await expect(page.getByText('T−24 h · Until event', { exact: true })).toBeVisible();
  await page.getByRole('navigation', { name: 'Demo forecast products' }).getByRole('link', { name: 'Solar Wind Plasma Density' }).click();
  await expect(page).toHaveURL('/demo/products/solar-wind-density?offset=-24');
  await expect(page.getByRole('heading', { name: 'Solar Wind Plasma Density', exact: true }).first()).toBeVisible();
  await page.getByRole('checkbox', { name: 'Show actual future values' }).uncheck();
  await expect(page.getByText('Hidden cm⁻³')).toBeVisible();
  await page.getByRole('button', { name: 'T0', exact: true }).click();
  await expect(page.getByText('T0 · Event begins', { exact: true })).toBeVisible();
  await expect(page.getByText('19 Jan 2026, 19:38 UTC · Simulated now')).toBeVisible();
  await page.getByRole('link', { name: 'Live observations', exact: true }).click();
  await expect(page).toHaveURL('/demo/live?offset=0');
  await expect(page.getByRole('heading', { name: 'Historical observations' })).toBeVisible();
  expect(requests).toEqual([]);
  await page.goto('/demo/products/solar-wind-density?offset=-24');
  await expect(page.getByRole('link', { name: 'Exit demo' })).toHaveAttribute('href', '/products/solar-wind-density');
  await page.getByRole('link', { name: 'Exit demo' }).click();
  await expect(page).toHaveURL('/products/solar-wind-density');
  await expect(page.getByRole('region', { name: 'Demo time controls' })).toHaveCount(0);
});

test('panel remains visible without covering content on mobile and keyboard changes time', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/demo?offset=-96');
  const panel = page.getByRole('region', { name: 'Demo time controls' });
  const slider = page.getByRole('slider', { name: 'Hours before event' });
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
  await page.goto('/demo/products/dst?offset=oops');
  await expect(page.getByRole('slider')).toHaveValue('-96');
  await expect(page.getByRole('heading', { name: 'Dst Index', exact: true })).toBeVisible();
  await expect(page.getByText('No demo release is available at this time.')).toBeVisible();
  await page.goto('/demo/not-a-page');
  await expect(page.getByRole('region', { name: 'Demo time controls' })).toHaveCount(0);
});

test('playback stops at T0 and can restart from the beginning', async ({ page }) => {
  await page.goto('/demo?offset=-1');
  await page.getByRole('button', { name: 'Play replay' }).click();
  await expect(page.getByRole('slider')).toHaveValue('0');
  await expect(page.getByRole('button', { name: 'Play replay' })).toBeVisible();
  await expect(page.getByText('T0 · Event begins', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Play replay' }).click();
  await expect(page.getByRole('slider')).toHaveValue('-96');
  await page.getByRole('button', { name: 'Pause replay' }).click();
  await expect(page.getByRole('button', { name: 'Play replay' })).toBeVisible();
});

test('other products replay probabilities, signed indices and partial model availability', async ({ page }) => {
  const operationalRequests: string[] = [];
  page.on('request', request => { if (request.url().includes('/api/v1/')) operationalRequests.push(request.url()); });
  await page.goto('/demo/products?offset=-48');
  await expect(page.getByRole('link', { name: /Geomagnetic Activity.*Explore demo forecast/ })).toBeVisible();
  await page.goto('/demo/products/geomagnetic-activity?offset=-48');
  await expect(page.getByRole('region', { name: 'Kp Index forecast and observations' })).toBeVisible();
  await expect(page.getByText('Median · q50', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'AP Ap Index' }).click();
  await expect(page.getByRole('region', { name: 'Ap Index forecast and observations' })).toBeVisible();
  await page.goto('/demo/products/dst?offset=-48');
  await expect(page.getByRole('region', { name: 'Dst Index forecast and observations' })).toBeVisible();
  await expect(page.getByText('-45', { exact: false }).first()).toBeVisible();
  await page.goto('/demo/products/hmf?offset=-48');
  await expect(page.getByRole('region', { name: 'Total HMF forecast and observations' })).toBeVisible();
  await page.getByRole('button', { name: 'BS Southward Bz' }).click();
  await expect(page.getByText('Southward Bz is not available in this release.')).toBeVisible();
  await page.goto('/demo/products/solar-radiation?offset=-48');
  await expect(page.getByRole('region', { name: 'F10.7 forecast and observations' })).toBeVisible();
  await page.getByRole('button', { name: 'S10 S10' }).click();
  await expect(page.getByText('S10 is not available in this release.')).toBeVisible();
  expect(operationalRequests).toEqual([]);
});


test('demo controls wait for hydration before accepting replay input', async ({ page }) => {
  let releaseScripts!: () => void;
  const scriptsReady = new Promise<void>(resolve => { releaseScripts = resolve; });
  await page.route('**/_next/**/*.js*', async route => {
    await scriptsReady;
    await route.continue();
  });
  await page.goto('/demo?offset=-96', { waitUntil: 'commit' });
  const slider = page.getByRole('slider', { name: 'Hours before event' });
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
