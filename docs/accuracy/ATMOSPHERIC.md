# Atmospheric density accuracy

Updated October 10, 2026.

**The current working model is JB2008.** The registry still selects
`models.atmospheric_density.model: JB2008` in
[models_registry.yaml](../../configs/models_registry.yaml).
DTM2020 integration is implemented but has not been activated: a suitable CNES
license and a provisioned runtime are required. Forecast ap is not enabled.

## Compared variants

| Variant | Inputs over the forecast horizon | Status |
|---|---|---|
| JB2008 | Observed solar drivers and derived DTC held fixed | Current working model; experimental control |
| DTM2020 Operational | Observed F10.7, its trailing 81-day mean, delayed Kp and trailing 24-hour mean Kp held fixed | Integration prepared; not activated |
| DTM2020 Operational + forecast ap | Forecast ap q50 converted to Kp; F10.7 and its background remain identical to fixed-input DTM | Experimental only |

The third variant forecasts **ap only**, not all inputs. Conversion from ap to
continuous Kp uses a table with linear interpolation. The adapter interpolates
ap first, then converts to Kp, applies a three-hour lag and a trailing 24-hour
Kp mean. Its ap model is a separately trained experimental model, not the
undated production ap bundle. Driver preparation is separate from the DTM
physical evaluator to support later forecast-input integration.

## Evaluation sample

- Reference: ESA GRACE-FO 1 accelerometer-derived density, DNS ACC v0003.
- Windows: January 1–14, June 1–14 and November 10–23, 2025; 42 days in total.
- Observations sampled every 30 minutes, at approximately 452–503 km altitude.
- 2,016 target observations. At each nonzero lead, all three variants are scored
  on the same 2,012 available pairs; four pairs lack ap forecast features.
- Storm subset: observed Kp at verification time >= 5; 288 pairs per lead.
  These are correlated observations, not 288 independent storms. Future Kp is
  used only to label evaluation subsets.
- Experimental ap q50 training uses 2010–2024 data with target availability
  strictly before December 1, 2024. No 2025 targets were used to fit that model.

These are retrospective density scores on the same observed satellite positions.
They do not measure orbit prediction accuracy or the accuracy of satellite drag-risk
categories. The windows have been inspected during research and are not a fresh
blind holdout. Revised archives and assumed data availability are used; this is
not a historical replay of live publication receipts.

## Mean absolute percentage error

MAPE = `100 × mean(abs(predicted_density / observed_density − 1))`.
Densities are in kg/m³; the table reports percentages. **The full sample includes
storms**, so these figures must not be described as quiet-condition accuracy.

| Lead | N per model | JB2008 | DTM fixed inputs | DTM + forecast ap q50 |
|---|---:|---:|---:|---:|
| +6 h | 2,012 | 27.59% | 21.11% | 18.92% |
| +12 h | 2,012 | 31.12% | 23.76% | 19.44% |
| +24 h | 2,012 | 36.26% | 25.76% | 19.37% |
| +48 h | 2,012 | 42.64% | 30.45% | 20.36% |

Fixed-input DTM already improves average error over the current JB2008 chain on
this sample. Forecast ap further improves average error, with a storm-specific
tradeoff shown below.

## Storm performance

Strong underestimation means predicted density is below 70% of observed density.
Its frequency is a percentage of observation pairs, **not MAPE** and not a
probability of satellite loss. Bias is the mean signed percentage error;
negative values indicate systematic underestimation.

| Metric on storm subset | Lead | JB2008 | DTM fixed inputs | DTM + forecast ap q50 |
|---|---|---:|---:|---:|
| MAPE | +24 h | 46.96% | 39.18% | 29.56% |
| MAPE | +48 h | 46.16% | 40.40% | 32.12% |
| Strong underestimation frequency | +24 h | 42.36% | 31.25% | 39.93% |
| Strong underestimation frequency | +48 h | 52.78% | 42.71% | 49.65% |
| Bias | +24 h | −0.97% | +0.86% | −19.28% |
| Bias | +48 h | −14.26% | −8.62% | −24.10% |

Adding forecast ap reduces storm MAPE but increases strong underestimation
relative to **fixed-input DTM**. At +24 h the frequency rises from 31.25% to
39.93%: +8.68 percentage points, approximately +28% relative. This is not a
comparison against JB2008; JB2008 has both higher MAPE and a slightly higher
strong-underestimation frequency than DTM + forecast ap at these horizons.

## Follow-up diagnostics

Replacing forecast ap with actual future ap, while retaining other inputs and
the same adapter, reduces storm strong-underestimation frequency to **8.33% at
+24 h** and **10.42% at +48 h**. Storm MAPE remains **27.23% / 27.42%**; full-sample
MAPE becomes **18.96% / 19.19%**. This supports forecast ap errors being a
substantial source of storm underestimation in this chain. It is an oracle
diagnostic using unavailable future information, not a deployable fourth model
or a guaranteed lower bound on density error.

Upper ap q90/q95 scenarios also reduce strong underestimation but increase
large overestimates. They are not density confidence bounds. A separate density
U95 calibration fitted on December 3–28, 2024 covered only **56.60%** of storm
observations at +24 h. That predominantly quiet calibration does not justify a
95% storm-coverage claim or operational risk probabilities.

>The intended next working model is **DTM2020 with fixed inputs**, after licensing
>and runtime provisioning. **JB2008 remains selected today.** Forecast ap requires
>further storm validation before activation.
