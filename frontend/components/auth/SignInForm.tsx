'use client'

import { useState } from 'react'
import { Input }  from '@/components/ui/Input'
import { Button } from '@/components/ui/Button'
import { doSignIn } from '@/lib/auth'
import { AuthCard } from './AuthCard'

interface SignInFormProps {
  onSuccess:  () => void
  onSignUp:   () => void
}

export function SignInForm({ onSuccess, onSignUp }: SignInFormProps) {
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [error,    setError]    = useState<string | null>(null)
  const [loading,  setLoading]  = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await doSignIn(email, password)
      onSuccess()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Sign in failed.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthCard title="Welcome back" subtitle="Sign in to your AmanAI account">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4" noValidate>
        <Input
          label="Email"
          type="email"
          autoComplete="email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
        />
        <Input
          label="Password"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
        />

        {error && (
          <p className="text-sm text-red-600 dark:text-red-400 rounded-lg bg-red-50 dark:bg-red-900/20 px-3 py-2" role="alert">
            {error}
          </p>
        )}

        <Button type="submit" loading={loading} className="w-full mt-1">
          Sign in
        </Button>
      </form>

      <p className="mt-5 text-center text-sm text-[var(--color-text-muted)]">
        Don&apos;t have an account?{' '}
        <button onClick={onSignUp} className="text-primary-600 dark:text-primary-400 font-medium hover:underline focus-visible:outline-none">
          Sign up
        </button>
      </p>
    </AuthCard>
  )
}
