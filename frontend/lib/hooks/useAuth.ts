'use client'

import { useState, useEffect, useCallback } from 'react'
import { getAuthUser, doSignOut, type AuthUser } from '@/lib/auth'
import { Hub } from 'aws-amplify/utils'

type AuthState = {
  user:     AuthUser | null
  loading:  boolean
  signOut:  () => Promise<void>
  refresh:  () => Promise<void>
}

export function useAuth(): AuthState {
  const [user,    setUser]    = useState<AuthUser | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    setLoading(true)
    const u = await getAuthUser()
    setUser(u)
    setLoading(false)
  }, [])

  useEffect(() => {
    refresh()

    const unsubscribe = Hub.listen('auth', ({ payload }: { payload: { event: string } }) => {
      if (['signedIn', 'signedOut', 'tokenRefresh'].includes(payload.event)) {
        refresh()
      }
    })
    return unsubscribe
  }, [refresh])

  const signOut = useCallback(async () => {
    await doSignOut()
    setUser(null)
  }, [])

  return { user, loading, signOut, refresh }
}
