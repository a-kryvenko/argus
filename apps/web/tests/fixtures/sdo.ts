import type { Page } from '@playwright/test';

export const sdoChannels = ['094', '131', '171', '193', '211', '304', '335', '1600', '1700'];
export const sdoFixture = () => Object.fromEntries(sdoChannels.map(channel => [channel, {
  asset_id: `2026103012_${channel}_${'a'.repeat(64)}`,
  urls: Object.fromEntries(['jpg', 'webp'].flatMap(format => [128, 256, 512, 1024, 2048, 4096].map(size => [
    format === 'jpg' && size === 4096 ? 'jpg.original' : `${format}.${size}`,
    `http://localhost:8333/argus-public/images/AIA/${Number(channel)}/202610301230/${format === 'jpg' && size === 4096 ? 'original' : size}.${format}`,
  ]))),
}]));

export async function mockSDO(page: Page) {
  await page.route('http://localhost:8333/argus-public/**', route => route.fulfill({ contentType: 'image/svg+xml',
    body: '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><rect width="256" height="256" fill="black"/><circle cx="128" cy="128" r="103" fill="#d87714"/></svg>' }));
}
