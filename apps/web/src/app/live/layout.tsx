import type { ReactNode } from "react";
import type { Metadata } from "next";

const origin = 'https://argussun.com';
const title = 'Live Solar Wind, Bz & Kp Index';
const description = 'Live solar wind speed, Bz/Bt magnetic field, proton density, Kp, Dst and F10.7 solar flux. Explore 30-day space weather charts and JSON data.';

export const metadata: Metadata = {
  metadataBase: new URL(origin),
  title,
  description,
  alternates: { canonical: '/live' },
  openGraph: {
    type: 'website', url: '/live', siteName: 'Argus SunWatch', locale: 'en_US',
    title: `${title} | Argus SunWatch`, description,
  },
  twitter: { card: 'summary_large_image', title: `${title} | Argus SunWatch`, description, images: ['/live/opengraph-image'] },
};

const dataset = {
  '@context': 'https://schema.org',
  '@type': 'Dataset',
  '@id': `${origin}/live#observations`,
  name: 'Argus SunWatch live space weather observations',
  description,
  url: `${origin}/live`,
  inLanguage: 'en',
  isAccessibleForFree: true,
  publisher: { '@type': 'Organization', name: 'Argus SunWatch', url: origin },
  isBasedOn: [
    { '@type': 'CreativeWork', name: 'NOAA SWPC observations', url: 'https://www.spaceweather.gov/' },
    { '@type': 'CreativeWork', name: 'WDC Kyoto Dst index', url: 'https://wdc.kugi.kyoto-u.ac.jp/dstdir/' },
  ],
  variableMeasured: [
    ['Solar wind speed', 'v', 'km/s'], ['Proton density', 'n', 'cm^-3'],
    ['GSM Bz magnetic field', 'bz', 'nT'], ['Total magnetic field Bt', 'bt', 'nT'],
    ['Planetary Kp index', 'kp', ''], ['Dst index', 'dst', 'nT'], ['F10.7 solar radio flux', 'f10_7', 'sfu'],
  ].map(([name, propertyID, unitText]) => ({ '@type': 'PropertyValue', name, propertyID, ...(unitText ? { unitText } : {}) })),
  distribution: [
    { '@type': 'DataDownload', name: 'Current observations', encodingFormat: 'application/json', contentUrl: `${origin}/api/v1/observations` },
    { '@type': 'DataDownload', name: 'Observation history', encodingFormat: 'application/json', contentUrl: `${origin}/api/v1/observations/history?observations=v,n,bz,bt,kp,dst&resolution=auto` },
  ],
};

export default function LiveLayout({ children }: { children: ReactNode }) {
  return <>
    <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(dataset).replace(/</g, '\\u003c') }} />
    {children}
  </>;
}
