import Link from "next/link";
import { products } from "../_config/products";

export const metadata = { title: "Products" };

export default function Products() {
    return (
        <main className="container color-default">
            <h1 className="heading">Products</h1>
            <div className="product-grid">
              {products.map(product => (
                <Link className="product-link" href={`/products/${product.slug}`} key={product.slug}>
                  <h2>{product.title}</h2>
                  <p>{product.description}</p>
                </Link>
              ))}
            </div>
        </main>
    )
}
