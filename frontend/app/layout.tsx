import type { Metadata } from 'next'
import './globals.css'
import { AmplifyProvider } from '@/components/providers/AmplifyProvider'
import { ThemeProvider }   from '@/components/providers/ThemeProvider'

export const metadata: Metadata = {
  title:       'AmanAI — NUST Bank Assistant',
  description: 'Your intelligent NUST Bank financial assistant, powered by AI.',
  icons: {
    icon: '/favicon.svg',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head />
      <body>
        <ThemeProvider>
          <AmplifyProvider>
            {children}
          </AmplifyProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
