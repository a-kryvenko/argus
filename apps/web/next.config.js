/** @type {import('next').NextConfig} */
const nextConfig = {
    output: "standalone",
    distDir: process.env.NEXT_DIST_DIR || ".next",

    async rewrites() {
        if (process.env.NODE_ENV !== "development") {
            return [];
        }

        return [
            {
                source: "/api/:path*",
                destination: "http://localhost:8000/api/:path*",
            },
        ];
    },
};

export default nextConfig;
