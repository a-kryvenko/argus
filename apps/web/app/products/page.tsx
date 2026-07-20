import Link from "next/link";
import { products } from "../_config/products";

export default function Products() {
    return (
        <div className="container color-default">
            <h1 className="text-center">Products</h1>
            <div className="product-grid">
              {products.map(product => (
                <Link className="product-link" href={`/products/${product.slug}`} key={product.slug}>
                  <h2>{product.title}</h2>
                  <p>{product.description}</p>
                </Link>
              ))}
            </div>
        </div>
    )
}
