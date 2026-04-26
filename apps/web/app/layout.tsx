import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Header from '../components/Header';

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
    <html lang="ru">
      <body className={`${inter.className} bg-zinc-950 text-zinc-100`}>
        <Header />
        {children}
      </body>
    </html>
  );
}