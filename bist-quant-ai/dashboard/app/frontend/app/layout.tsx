import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'BIST Daily Portfolio Dashboard',
  description: 'Daily portfolio monitoring dashboard',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-dark-950 text-white antialiased">{children}</body>
    </html>
  )
}
