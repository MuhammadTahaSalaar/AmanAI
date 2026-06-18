'use client'

import { useEffect } from 'react'
import '@/lib/amplify'

export function AmplifyProvider({ children }: { children: React.ReactNode }) {
  // Amplify is configured via the side-effect import above.
  // This component exists so Amplify initialisation only runs client-side.
  useEffect(() => {
    // no-op: amplify.ts ran on import
  }, [])

  return <>{children}</>
}
