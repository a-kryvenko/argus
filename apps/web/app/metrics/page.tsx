import Link from "next/link";
import { products } from "../_config/products";

export const metadata = { title: "Metrics" };

export default function Metrics() {
    return (
        <main className="container color-default">
            <h1 className="heading">Metrics</h1>
            <div className="product-grid">
              {products.map(product => (
                <Link className="product-link" href={`/metrics/${product.slug}`} key={product.slug}>
                  <h2>{product.title}</h2>
                  <p>Forecast quality by lead hour</p>
                </Link>
              ))}
            </div>
        </main>
    )
}
