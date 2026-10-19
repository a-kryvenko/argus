import type { MetadataRoute } from 'next';

export default function sitemap(): MetadataRoute.Sitemap {
  return ['/', '/live', '/forecast', '/metrics', '/help'].map(path => ({ url: `https://argussun.com${path}` }));
}
