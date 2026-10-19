import { ImageResponse } from 'next/og';

export const alt = 'Argus SunWatch — live solar wind and geomagnetic observations';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default function Image() {
  return new ImageResponse(
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: 64, background: '#10171f', color: '#e0edfa' }}>
      <div style={{ display: 'flex', fontSize: 24, letterSpacing: 5, color: '#a5bdd0' }}>ARGUS SUNWATCH</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div style={{ fontSize: 72, fontWeight: 700 }}>Live space weather</div>
        <div style={{ fontSize: 30, color: '#a5bdd0' }}>Solar wind · Bz / Bt · Kp · Dst · F10.7</div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '2px solid #304656', paddingTop: 26, fontSize: 22 }}>
        <span style={{ color: '#81c7b7' }}>Observations · History · JSON API</span>
        <span>argussun.com/live</span>
      </div>
    </div>,
    size,
  );
}
