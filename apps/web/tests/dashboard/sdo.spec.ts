import { test, expect, type Page } from '@playwright/test';
import { mockSDO, sdoChannels, sdoFixture } from '../fixtures/sdo';
import { captureScreenshot } from './screenshot';

async function mock(page: Page, assets = sdoFixture()) {
  await mockSDO(page);
  await page.route('**/api/v1/summary', route => route.fulfill({ json: { success: true, error: null, data: {
    generated_at: '2026-10-30T12:00:00Z', cme_events: [], cme_events_available: true, sdo_images: assets,
    observations: Object.fromEntries(['plasma', 'geo', 'hmf', 'radio'].map(group => [group, { title: group, items: [] }])),
  } } }));
}

test('nine AIA thumbnails stay in one row on desktop and mobile with JPEG fallback', async ({ page }) => {
  await mock(page);
  await page.goto('/');
  const strip = page.getByRole('group', { name: 'SDO AIA channels' });
  await expect(strip.getByRole('button')).toHaveCount(9);
  await expect(strip.getByRole('button').first()).toBeEnabled();
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    const positions = await strip.getByRole('button').evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().y));
    expect(new Set(positions).size).toBe(1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  for (const channel of sdoChannels) {
    const picture = strip.getByRole('button', { name: `Open AIA ${channel} Å` }).locator('picture');
    await expect(picture.locator('source')).toHaveAttribute('type', 'image/webp');
    await expect(picture.locator('source')).toHaveAttribute('srcset', /_128.webp 128w, .*_256.webp 256w, .*_512.webp 512w/);
    await expect(picture.locator('img')).toHaveAttribute('srcset', /_128.jpg 128w, .*_256.jpg 256w, .*_512.jpg 512w/);
  }
  await captureScreenshot(page, { path: 'test-results/sdo-mobile.png', fullPage: true });
});

test('large images and direct NASA movies load only after user actions; Escape restores focus', async ({ page }) => {
  const requested: string[] = [];
  page.on('request', request => requested.push(request.url()));
  await page.route('**/api/v1/sdo/movies/**', () => { throw new Error('Movie must never pass through our API'); });
  await mock(page);
  await page.route('https://sdo.gsfc.nasa.gov/assets/img/latest/mpeg/*', route => route.abort());
  await page.goto('/');
  const button = page.getByRole('button', { name: 'Open AIA 131 Å' });
  await expect(button).toBeEnabled();
  expect(requested.some(url => /_(1024|2048|4096)\.(webp|jpg)/.test(url))).toBe(false);
  expect(requested.some(url => url.endsWith('.mp4'))).toBe(false);
  await button.click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('heading', { name: 'SDO · AIA 131 Å' })).toBeVisible();
  await expect.poll(() => requested.some(url => /_(1024|2048|4096)\.webp/.test(url))).toBe(true);
  await expect(dialog.locator('img')).toHaveAttribute('srcset', /_4096.jpg 4096w/);
  expect(requested.some(url => url.endsWith('.mp4'))).toBe(false);
  await captureScreenshot(page, { path: 'test-results/sdo-viewer.png', fullPage: true });
  await dialog.getByRole('button', { name: 'Play 48h' }).click();
  await expect.poll(() => requested.some(url => url === 'https://sdo.gsfc.nasa.gov/assets/img/latest/mpeg/latest_1024_0131.mp4')).toBe(true);
  await expect(dialog.getByRole('alert')).toContainText('Video unavailable.');
  await expect(dialog.getByRole('status')).toHaveCount(0);
  await dialog.getByRole('button', { name: 'Image', exact: true }).click();
  await expect(dialog.locator('video')).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(button).toBeFocused();
});

test('keyboard arrows cycle large images through all channels and wrap in both directions', async ({ page }) => {
  await mock(page);
  await page.route('https://sdo.gsfc.nasa.gov/assets/img/latest/mpeg/*', route => route.abort());
  await page.goto('/');
  const opener = page.getByRole('button', { name: 'Open AIA 094 Å' });
  await opener.click();
  const dialog = page.getByRole('dialog');
  const expectChannel = async (channel: string) => {
    await expect(dialog.getByRole('heading')).toHaveText(`SDO · AIA ${channel} Å`);
    await expect(dialog.locator('img')).toHaveAttribute('alt', `Sun in SDO AIA ${channel} Å`);
  };
  await page.keyboard.press('ArrowLeft');
  await expectChannel('1700');
  await page.keyboard.press('ArrowRight');
  await expectChannel('094');
  for (const channel of [...sdoChannels.slice(1), '094']) {
    await page.keyboard.press('ArrowRight');
    await expectChannel(channel);
  }
  for (const channel of [...sdoChannels.slice(1).reverse(), '094']) {
    await page.keyboard.press('ArrowLeft');
    await expectChannel(channel);
  }
  await expect(dialog.getByRole('button', { name: 'Play 48h' })).toBeFocused();
  await dialog.getByRole('button', { name: 'Play 48h' }).click();
  await page.keyboard.press('ArrowRight');
  await expect(dialog.getByRole('heading')).toHaveText('SDO · AIA 094 Å');
  await dialog.getByRole('button', { name: 'Image', exact: true }).click();
  await page.keyboard.press('ArrowLeft');
  await expectChannel('1700');
  await page.keyboard.press('Escape');
  await expect(opener).toBeFocused();
});

test('movies load directly from NASA with one loader, start automatically, and respect a manual pause', async ({ page }) => {
  await mock(page);
  let release!: () => void;
  const ready = new Promise<void>(resolve => { release = resolve; });
  await page.route('https://sdo.gsfc.nasa.gov/assets/img/latest/mpeg/*', async route => {
    await ready;
    await route.fulfill({ path: 'tests/fixtures/sdo.mp4', contentType: 'video/mp4' });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Open AIA 131 Å' }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('button', { name: 'Play 48h' }).click();
  const video = dialog.locator('video');
  await expect(dialog.getByRole('status')).toHaveText('Loading video…');
  await expect(dialog.getByRole('status')).toHaveCount(1);
  await expect(video).toBeHidden();
  expect(await video.evaluate((node: HTMLVideoElement) => node.controls)).toBe(false);
  release();
  await expect.poll(() => video.evaluate((node: HTMLVideoElement) => !node.paused && node.currentTime > 0)).toBe(true);
  await expect(video.locator('source')).toHaveAttribute('src', 'https://sdo.gsfc.nasa.gov/assets/img/latest/mpeg/latest_1024_0131.mp4');
  await expect(video).toBeVisible();
  expect(await video.evaluate((node: HTMLVideoElement) => node.controls)).toBe(true);
  await expect(dialog.getByRole('status')).toHaveCount(0);
  expect(await video.evaluate((node: HTMLVideoElement) => node.muted)).toBe(true);
  await video.dispatchEvent('waiting');
  await expect(dialog.getByRole('status')).toHaveCount(0);
  await expect(video).toBeVisible();
  expect(await video.evaluate((node: HTMLVideoElement) => node.controls)).toBe(true);
  await video.dispatchEvent('playing');
  await video.evaluate((node: HTMLVideoElement) => {
    node.pause();
    node.dispatchEvent(new Event('canplaythrough'));
  });
  expect(await video.evaluate((node: HTMLVideoElement) => node.paused)).toBe(true);
  await expect(dialog.getByRole('status')).toHaveCount(0);
  await dialog.getByRole('button', { name: 'Image', exact: true }).click();
  await dialog.getByRole('button', { name: 'Play 48h' }).click();
  await expect.poll(() => video.evaluate((node: HTMLVideoElement) => !node.paused && node.currentTime > 0)).toBe(true);
});

test('fullscreen looping and buffering keep video controls and visibility unchanged', async ({ page }) => {
  await mock(page);
  await page.route('https://sdo.gsfc.nasa.gov/assets/img/latest/mpeg/*', route =>
    route.fulfill({ path: 'tests/fixtures/sdo.mp4', contentType: 'video/mp4' }));
  await page.goto('/');
  await page.getByRole('button', { name: 'Open AIA 131 Å' }).click();
  await page.getByRole('button', { name: 'Play 48h' }).click();
  const video = page.locator('video');
  await expect.poll(() => video.evaluate((node: HTMLVideoElement) => !node.paused && node.currentTime > 0)).toBe(true);
  await expect(video).toBeVisible();
  await video.evaluate(async (node: HTMLVideoElement) => {
    Object.assign(window, { movieControlChanges: 0 });
    new MutationObserver(records => {
      const state = window as unknown as { movieControlChanges: number };
      state.movieControlChanges += records.length;
    }).observe(node, { attributes: true, attributeFilter: ['controls', 'class'] });
    await node.requestFullscreen();
    node.currentTime = node.duration - 0.2;
  });
  await expect.poll(() => video.evaluate((node: HTMLVideoElement) => node.currentTime < 1)).toBe(true);
  await video.dispatchEvent('waiting');
  await expect(page.locator('[role="status"]')).toHaveCount(0);
  expect(await video.evaluate((node: HTMLVideoElement) => node.controls)).toBe(true);
  await expect(video).toBeVisible();
  await video.dispatchEvent('playing');
  await expect(page.locator('[role="status"]')).toHaveCount(0);
  expect(await video.evaluate(node => document.fullscreenElement === node)).toBe(true);
  expect(await page.evaluate(() => (window as unknown as { movieControlChanges: number }).movieControlChanges)).toBe(0);
  await page.evaluate(() => document.exitFullscreen());
});

test('a browser autoplay rejection leaves a working play button instead of a loader', async ({ page }) => {
  await mock(page);
  await page.addInitScript(() => {
    const play = HTMLMediaElement.prototype.play;
    let blocked = true;
    HTMLMediaElement.prototype.play = function () {
      if (blocked) {
        blocked = false;
        return Promise.reject(new DOMException('Autoplay blocked', 'NotAllowedError'));
      }
      return play.call(this);
    };
  });
  await page.route('https://sdo.gsfc.nasa.gov/assets/img/latest/mpeg/*', route =>
    route.fulfill({ path: 'tests/fixtures/sdo.mp4', contentType: 'video/mp4' }));
  await page.goto('/');
  await page.getByRole('button', { name: 'Open AIA 131 Å' }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('button', { name: 'Play 48h' }).click();
  await expect(dialog.getByRole('button', { name: 'Play video', exact: true })).toBeVisible();
  await expect(dialog.getByRole('status')).toHaveCount(0);
  await dialog.getByRole('button', { name: 'Play video', exact: true }).click();
  await expect.poll(() => dialog.locator('video').evaluate((node: HTMLVideoElement) => !node.paused && node.currentTime > 0)).toBe(true);
  await expect(dialog.getByRole('button', { name: 'Play video', exact: true })).toHaveCount(0);
});

test('summary polling pauses throughout video download and viewing, then resumes after closing', async ({ page }) => {
  await page.clock.install();
  let reads = 0;
  page.on('request', request => { if (new URL(request.url()).pathname === '/api/v1/summary') reads++; });
  await mock(page);
  let release!: () => void;
  const ready = new Promise<void>(resolve => { release = resolve; });
  await page.route('https://sdo.gsfc.nasa.gov/assets/img/latest/mpeg/*', async route => {
    await ready;
    await route.fulfill({ path: 'tests/fixtures/sdo.mp4', contentType: 'video/mp4' });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Open AIA 131 Å' }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('button', { name: 'Play 48h' }).click();
  await expect(dialog.getByRole('status')).toBeVisible();
  await page.clock.fastForward(10 * 60 * 1000);
  expect(reads).toBe(1);
  await expect(dialog).toBeVisible();
  release();
  await expect.poll(() => dialog.locator('video').evaluate((node: HTMLVideoElement) => !node.paused)).toBe(true);
  await page.clock.fastForward(10 * 60 * 1000);
  expect(reads).toBe(1);
  await expect(dialog.locator('video')).toBeVisible();
  await page.keyboard.press('Escape');
  await page.clock.fastForward(5 * 60 * 1000);
  await expect.poll(() => reads).toBe(2);
});

test('one missing SDO channel hides the whole block without hiding observations', async ({ page }) => {
  const assets = sdoFixture();
  delete assets['094'];
  await mock(page, assets);
  await page.goto('/');
  await expect(page.getByRole('region', { name: 'plasma', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'SDO AIA', exact: true })).toHaveCount(0);
});

test('a failed thumbnail hides the whole SDO block', async ({ page }) => {
  await mock(page);
  await page.route('**/api/v1/sdo/images/*_131_*', route => route.fulfill({ status: 404 }));
  const failed = page.waitForResponse(response => response.url().includes('/sdo/images/') && response.status() === 404);
  await page.goto('/');
  await failed;
  await expect(page.getByRole('region', { name: 'plasma', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'SDO AIA', exact: true })).toHaveCount(0);
});

test('an open page periodically discovers new hourly images', async ({ page }) => {
  await page.clock.install();
  let reads = 0;
  const requests: string[] = [];
  page.on('request', request => {
    const path = new URL(request.url()).pathname;
    if (path === '/api/v1/summary') reads++;
    if (path.startsWith('/api/v1/')) requests.push(path);
  });
  await mock(page);
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Open AIA 094 Å' })).toBeEnabled();
  expect(reads).toBe(1);
  expect(requests.filter(path => !path.startsWith('/api/v1/sdo/images/'))).toEqual(['/api/v1/summary']);
  await page.clock.fastForward(5 * 60 * 1000);
  await expect.poll(() => reads).toBe(2);
  await expect(page.getByRole('button', { name: 'Open AIA 094 Å' })).toBeEnabled();
});
