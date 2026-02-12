import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Quant AI — Institutional-Grade Trading Intelligence",
  description:
    "AI-powered multi-agent trading platform bringing institutional-grade quantitative strategies to retail investors. 34+ factor models, regime detection, and autonomous portfolio management for emerging markets.",
  keywords: [
    "quantitative trading",
    "AI trading",
    "factor investing",
    "emerging markets",
    "BIST",
    "portfolio management",
    "algorithmic trading",
  ],
  openGraph: {
    title: "Quant AI — Democratizing Institutional Finance",
    description:
      "Multi-agent AI platform with 34+ proven factor models. 102% CAGR backtested performance on BIST.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
