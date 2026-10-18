import type { DemoBundle } from '../../src/app/demo/_lib/replay';

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
    density_scenario: densityFixture(),
    event_releases: Array.from({ length: 169 }, (_, index) => {
      const hour = index - 96;
      return { issue_time: iso(hour), events: hour < -21 ? [] : [{ event_id: 'CME synthetic',
        status: 'predicted_hit', available_at: hour < -11 ? iso(-21) : iso(-11),
        arrival_at: hour < -11 ? '2026-01-20T02:18:00Z' : '2026-01-20T01:07:00Z' }] };
    }),
    event: { name: 'Synthetic browser test event', starts_at: '2026-01-19T19:38:00Z',
      source_url: 'https://example.com/test', description: 'Synthetic test fixture' },
    observation_source: 'Synthetic test fixture', input_source: 'Synthetic test fixture', models: {},
    observations: Array.from({ length: 289 }, (_, index) => ({ time: iso(index - 120), values: base })),
    releases: Object.fromEntries(definitions.map(({ target, variables }) => [target,
      Array.from({ length: 169 }, (_, index) => ({
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

export function densityFixture(): NonNullable<DemoBundle['density_scenario']> {
  const start = Date.parse('2026-01-19T19:00Z');
  const iso = (hour: number) => new Date(start + hour * 3600000).toISOString();
  const variants = ['jb2008_fixed', 'jb2008_forecast', 'dtm2020_fixed', 'dtm2020_forecast'] as const;
  const ratios = { jb2008_fixed: 1, jb2008_forecast: 3, dtm2020_fixed: 5, dtm2020_forecast: 6 };
  const levels = { jb2008_fixed: 'within_reference', jb2008_forecast: 'review', dtm2020_fixed: 'high', dtm2020_forecast: 'high' } as const;
  return {
    schema_version: 1, status: 'synthetic_test_only', spacecraft: 'GRACE-C',
    reference: { method: 'test_only', density_kg_m3: 1e-12, start: iso(-168), end_exclusive: iso(-96), samples: 144, expected_samples: 144, altitude_range_km: [450, 480], availability: 'synthetic' },
    policy: { review_ratio: 2, high_ratio: 4 }, variants: [...variants],
    releases: Object.fromEntries(variants.map(variant => [variant, Array.from({ length: 169 }, (_, index) => ({
      issue_time: iso(index - 96), attention: levels[variant], removed_warning_samples: 0,
      intervals: variant === 'jb2008_fixed' ? [] : [{ start: iso(index - 95), end: iso(index - 94), attention: levels[variant], sample_count: 2, peak_ratio: ratios[variant] }],
      predictions: [1, 2].map(lead => ({ time: iso(index - 96 + lead), density_kg_m3: ratios[variant] * 1e-12, ratio: ratios[variant], attention: levels[variant] })),
    })).filter((_, index) => variant !== 'dtm2020_forecast' || index !== 1)])) as NonNullable<DemoBundle['density_scenario']>['releases'],
    observations: Array.from({ length: 217 }, (_, index) => ({ time: iso(index - 120), density_kg_m3: 3e-12, attention: 'review' as const })),
    scores: variants.flatMap(variant => [2, 4].map(threshold => ({ variant, threshold_ratio: threshold, lead_hours: 24, requested: 169, usable: 169, excluded: 0, true_positive: 10, false_positive: 3, false_negative: 4, true_negative: 152, precision: 10 / 13, recall: 10 / 14 }))),
    journal: variants.flatMap(variant => Array.from({ length: 169 }, (_, index) => ({ variant, issue_time: iso(index - 96), attention: levels[variant], warning_intervals: 0, usable_samples: 2, missing_samples: 0, overlapping_samples: 1, changed_samples: 0, removed_warning_samples: 0, appeared_warning_samples: 0 }))),
    density_metrics: [], events: [], limitations: ['Synthetic browser-test data.'],
  };
}
