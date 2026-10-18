import { createLucideIcon } from 'lucide-react';

export default createLucideIcon({
  name: 'drag-risk',
  size: 24,
  node: [
    ['path', { d: 'M20.341 6.484A10 10 0 1 1 13.74 2.152', key: 'orbit' }],
    ['circle', { cx: '12', cy: '12', r: '3', key: 'center' }],
    ['circle', { cx: '19', cy: '5', r: '2', key: 'satellite' }],
  ],
});
