import { notFound } from "next/navigation";

import { productsBySlug } from "../../_config/products";
import MetricsProduct from "../_components/MetricsProduct";

export default async function MetricsPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const product = productsBySlug[slug];
  if (!product) notFound();
  return <MetricsProduct product={product} />;
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const product = productsBySlug[slug];
  if (!product) return { title: "Not found" };
  return { title: product.title + " Metrics", description: product.description };
}
