import type { DemoBundle } from '../../app/demo/_lib/replay';

// Synthetic values are confined to browser tests, never shipped as demo content.
export function demoFixture(): DemoBundle {
  const start = Date.parse('2026-01-19T19:00:00Z');
  const iso = (hour: number) => new Date(start + hour * 3600000).toISOString();
  const definitions = [
    { target: 'solar-wind-speed', variables: ['v'] },
    { target: 'solar-wind-density', variables: ['n'] },
    { target: 'geomagnetic-activity', variables: ['kp', 'ap'] },
    { target: 'dst', variables: ['dst'] },
    { target: 'hmf', variables: ['bt'] },
    { target: 'solar-radiation', variables: ['f10_7'] },
  ];
  const base: Record<string, number> = { v: 400, n: 5, kp: 5, ap: 30, dst: -45, bt: 8, f10_7: 140 };
  return {
    schema_version: 1, version: 'browser-test-only', generated_at: iso(200),
    event: { name: 'Synthetic browser test event', starts_at: '2026-01-19T19:38:00Z',
      source_url: 'https://example.com/test', description: 'Synthetic test fixture' },
    observation_source: 'Synthetic test fixture', input_source: 'Synthetic test fixture', models: {},
    observations: Array.from({ length: 217 }, (_, index) => ({ time: iso(index - 120), values: base })),
    releases: Object.fromEntries(definitions.map(({ target, variables }) => [target,
      Array.from({ length: 97 }, (_, index) => ({
        target, issue_time: iso(index - 96), horizon_hours: 96, available_variables: variables,
        predictions: Array.from({ length: 96 }, (_, lead) => ({
          valid_time: iso(index - 95 + lead), lead_hours: lead + 1,
          variables: Object.fromEntries(variables.map(variable => [variable, {
            continuous: ['kp', 'bt'].includes(variable) ? null : { q10: base[variable] - 2, q50: base[variable], q90: base[variable] + 2 },
            binary: ['kp', 'v'].includes(variable) ? [{ threshold: variable === 'v' ? 450 : 5, probability: .25 }] : [],
          }])),
        })),
      })).filter((_, index) => target !== 'dst' || index !== 0),
    ])),
  };
}
