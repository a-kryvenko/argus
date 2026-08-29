/** @type {import('next').NextConfig} */
const nextConfig = {
    output: "standalone",

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
