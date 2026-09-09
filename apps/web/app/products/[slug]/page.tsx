import { notFound } from "next/navigation";

import { productsBySlug } from "../../_config/products";
import ForecastProduct from "../_components/ForecastProduct";

export default async function ProductPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const product = productsBySlug[slug];
  if (!product) notFound();
  return <ForecastProduct product={product} />;
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const product = productsBySlug[slug];
  if (!product) return { title: "Not found" };
  return { title: product.title, description: product.description };
}
