'use client';

import { useRef, useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { Play, X } from 'lucide-react';
import styles from './sdo.module.css';

type SolarImage = { channel: string; asset_id: string };
const channels = ['094', '131', '171', '193', '211', '304', '335', '1600', '1700'];
const small = [128, 256, 512];
const large = [1024, 2048, 4096];
const url = (image: SolarImage, size: number, format: string) => `/api/v1/sdo/images/${image.asset_id}_${size}.${format}`;

function Picture({ image, full = false, onUnavailable }: { image: SolarImage; full?: boolean; onUnavailable?: () => void }) {
  const [failed, setFailed] = useState(false);
  if (failed) return <span className={styles.missing}>Image unavailable</span>;
  const sizes = full ? large : small;
  const display = full ? '(max-aspect-ratio: 1/1) calc(100vw - 24px), calc(100dvh - 80px)' : '(max-width: 1000px) 96px, 140px';
  const srcSet = (format: string) => sizes.map(size => `${url(image, size, format)} ${size}w`).join(', ');
  return <picture className={styles.picture}>
    <source type="image/webp" srcSet={srcSet('webp')} sizes={display} />
    <img src={url(image, full ? 1024 : 256, 'jpg')} srcSet={srcSet('jpg')} sizes={display}
      width={full ? 4096 : 256} height={full ? 4096 : 256} decoding="async"
      alt={`Sun in SDO AIA ${image.channel} Å`} onError={() => { setFailed(true); onUnavailable?.(); }} />
  </picture>;
}

function Viewer({ image }: { image: SolarImage }) {
  const [playing, setPlaying] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const code = image.channel.padStart(4, '0');
  return <>
    <div className={styles.viewerHeader}>
      <div><Dialog.Title>SDO · AIA {image.channel} Å</Dialog.Title>
        <Dialog.Description>NASA / SDO{playing ? ' · Live 48-hour movie from NASA' : ''}</Dialog.Description></div>
      <div className={styles.actions}>
        <button onClick={() => { setPlaying(value => !value); setVideoError(false); }}>{playing ? 'Image' : <><Play size={14} aria-hidden="true" />Play 48h</>}</button>
        <Dialog.Close className={styles.close} aria-label="Close SDO viewer"><X size={20} aria-hidden="true" /></Dialog.Close>
      </div>
    </div>
    <div className={styles.viewerBody}>
      {playing ? <div className={styles.movie}>
        <video controls autoPlay loop playsInline preload="none" onError={() => setVideoError(true)}
          poster={url(image, 1024, 'jpg')} aria-label={`SDO AIA ${image.channel} Å: latest 48 hours`}>
          <source src={`https://sdo.gsfc.nasa.gov/assets/img/latest/mpeg/latest_1024_${code}.mp4`} type="video/mp4" />
        </video>
        {videoError && <p>Video unavailable. <a href={`https://sdo.gsfc.nasa.gov/data/latest48.php?q=${code}`} target="_blank" rel="noreferrer">Open on NASA</a></p>}
      </div> : <Picture key={image.asset_id} image={image} full />}
    </div>
  </>;
}

export default function SDOChannels({ assets }: { assets: Record<string, string> }) {
  const [selected, setSelected] = useState<SolarImage | null>(null);
  const [unavailable, setUnavailable] = useState<string | null>(null);
  const opener = useRef<HTMLButtonElement | null>(null);
  const images = channels.flatMap(channel => {
    const asset_id = assets[channel];
    return asset_id ? [{ channel, asset_id }] : [];
  });
  if (images.length !== channels.length || images.some(image => image.asset_id === unavailable)) return null;
  return <section className={styles.section} aria-label="SDO AIA">
    <div className={styles.strip} role="group" aria-label="SDO AIA channels">
      {images.map(image => {
        return <button key={image.channel} className={styles.channel} aria-label={`Open AIA ${image.channel} Å`}
          onClick={event => { opener.current = event.currentTarget; setSelected(image); }}>
          <span className={styles.thumbnail}><Picture key={image.asset_id} image={image} onUnavailable={() => setUnavailable(image.asset_id)} />
            <span className={styles.channelName} aria-hidden="true">{image.channel} Å</span>
          </span>
        </button>;
      })}
    </div>
    <Dialog.Root open={selected !== null} onOpenChange={open => { if (!open) setSelected(null); }}>
      <Dialog.Portal><Dialog.Overlay className={styles.overlay} />
        <Dialog.Content className={styles.viewer} onCloseAutoFocus={event => {
          event.preventDefault();
          opener.current?.focus();
        }}>
          {selected && <Viewer key={selected.asset_id} image={selected} />}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  </section>;
}
