export type ProductVariable = {
  key: string;
  label: string;
  unit: string;
  quantile: boolean;
  thresholds: Array<{ value: number; label: string }>;
};

export type ProductConfig = {
  slug: string;
  apiTarget: string;
  visibility: "public" | "private";
  title: string;
  description: string;
  variables: ProductVariable[];
};

export const products: ProductConfig[] = [
  {
    slug: "solar-wind-speed",
    apiTarget: "solar-wind-speed",
    visibility: "public",
    title: "Solar Wind Speed",
    description: "Quantile forecast and probabilities of high-speed solar wind.",
    variables: [{
      key: "v", label: "Solar Wind Speed", unit: "km/s", quantile: true,
      thresholds: [450, 500, 600].map(value => ({ value, label: `V ≥ ${value} km/s` })),
    }],
  },
  {
    slug: "solar-wind-density",
    apiTarget: "solar-wind-density",
    visibility: "public",
    title: "Solar Wind Plasma Density",
    description: "Quantile forecast of proton number density in the solar wind.",
    variables: [{ key: "n", label: "Plasma Density", unit: "cm⁻³", quantile: true, thresholds: [] }],
  },
  {
    slug: "hmf",
    apiTarget: "hmf",
    visibility: "private",
    title: "Heliospheric Magnetic Field",
    description: "Threshold probabilities for total field strength and southward Bz.",
    variables: [
      {
        key: "bt", label: "Total HMF", unit: "nT", quantile: false,
        thresholds: [5, 10, 15].map(value => ({ value, label: `Bt ≥ ${value} nT` })),
      },
      {
        key: "southward_bz", label: "Southward Bz", unit: "nT", quantile: false,
        thresholds: [5, 10, 15].map(value => ({ value, label: `Southward Bz ≥ ${value} nT` })),
      },
    ],
  },
  {
    slug: "solar-radiation",
    apiTarget: "solar-radiation",
    visibility: "private",
    title: "Solar Radiation Indices",
    description: "Joint forecast of the F10.7, S10, M10 and Y10 solar indices.",
    variables: [
      { key: "f10_7", label: "F10.7", unit: "sfu", quantile: true, thresholds: [] },
      { key: "s10", label: "S10", unit: "index", quantile: true, thresholds: [] },
      { key: "m10", label: "M10", unit: "index", quantile: true, thresholds: [] },
      { key: "y10", label: "Y10", unit: "index", quantile: true, thresholds: [] },
    ],
  },
  {
    slug: "geomagnetic-activity",
    apiTarget: "geomagnetic-activity",
    visibility: "public",
    title: "Geomagnetic Activity",
    description: "Kp threshold probabilities and the related Ap quantile forecast.",
    variables: [
      {
        key: "kp", label: "Kp Index", unit: "index", quantile: false,
        thresholds: [4, 5, 6].map(value => ({ value, label: `Kp ≥ ${value}` })),
      },
      { key: "ap", label: "Ap Index", unit: "index", quantile: true, thresholds: [] },
    ],
  },
  {
    slug: "dst",
    apiTarget: "dst",
    visibility: "public",
    title: "Dst Index",
    description: "Quantile forecast of the storm-time disturbance index.",
    variables: [{ key: "dst", label: "Dst Index", unit: "nT", quantile: true, thresholds: [] }],
  },
];

export const productsBySlug = Object.fromEntries(products.map(product => [product.slug, product]));

export function productApiPath(product: ProductConfig, suffix = "") {
  return `/api/v1/${product.visibility}/forecasts/${product.apiTarget}${suffix}`;
}
