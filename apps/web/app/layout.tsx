import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Header from './_components/Header';
import Footer from './_components/Footer';

const inter = Inter({ subsets: ['latin', 'cyrillic'] });

export const metadata: Metadata = {
  title: 'Argus SunWatch',
  description: 'Solar Wind & Geomagnetic Impact Forecasting',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="d-flex flex-column justify-content-between">
        <div>
            <Header />
          {children}
        </div>
        <Footer />
      </body>
    </html>
  );
}