import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import SiteChrome from "./_components/SiteChrome";

const font = localFont({
  src: "../fonts/GeistVF.woff",
  display: "swap",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: {
    default: "Forecast | Argus SunWatch",
    template: "%s | Argus SunWatch",
  },
  icons: {
    icon: [
      { url: "/sun.svg", type: "image/svg+xml" },
      { url: "/favicon.ico", sizes: "16x16 32x32 48x48" },
    ],
    apple: "/apple-touch-icon.png",
  },
  description: "Solar wind observations, geomagnetic indices and forecasts.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={`${font.className} d-flex flex-column justify-content-between`}
      >
        <SiteChrome>{children}</SiteChrome>
      </body>
    </html>
  );
}
