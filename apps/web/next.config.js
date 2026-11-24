/** @type {import('next').NextConfig} */
const nextConfig = {
    output: "standalone",
    distDir: process.env.NEXT_DIST_DIR || ".next",

    async redirects() {
        return [
            { source: "/products/:path*", destination: "/forecast/:path*", permanent: true },
            { source: "/demo/products/:path*", destination: "/demo/forecast/:path*", permanent: true },
        ];
    },

    async rewrites() {
        if (process.env.NODE_ENV !== "development") {
            return [];
        }

        return [
            {
                source: "/api/v1/dashboard/:path*",
                destination: "http://127.0.0.1:8000/api/v1/dashboard/:path*",
            },
            {
                source: "/api/v1/:path*",
                destination: "http://127.0.0.1:8003/api/v1/:path*",
            },
        ];
    },
};

export default nextConfig;
