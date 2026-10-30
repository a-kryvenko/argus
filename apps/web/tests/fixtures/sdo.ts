import type { Page } from '@playwright/test';

export const sdoChannels = ['094', '131', '171', '193', '211', '304', '335', '1600', '1700'];
export const sdoFixture = (): Record<string, string> => Object.fromEntries(sdoChannels.map(channel =>
  [channel, `2026103012_${channel}_${'a'.repeat(64)}`]));

export async function mockSDO(page: Page) {
  await page.route('**/api/v1/sdo/images/*', route => route.fulfill({ contentType: 'image/svg+xml',
    body: '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><rect width="256" height="256" fill="black"/><circle cx="128" cy="128" r="103" fill="#d87714"/></svg>' }));
}
