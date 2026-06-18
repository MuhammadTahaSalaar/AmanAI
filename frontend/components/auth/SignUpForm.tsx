'use client'

import { useState } from 'react'
import { Input }  from '@/components/ui/Input'
import { Button } from '@/components/ui/Button'
import { doSignUp, doConfirmSignUp } from '@/lib/auth'
import { AuthCard } from './AuthCard'

interface SignUpFormProps {
  onSuccess: () => void
  onSignIn:  () => void
}

export function SignUpForm({ onSuccess, onSignIn }: SignUpFormProps) {
  const [step,     setStep]     = useState<'register' | 'confirm'>('register')
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [code,     setCode]     = useState('')
  const [error,    setError]    = useState<string | null>(null)
  const [loading,  setLoading]  = useState(false)

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await doSignUp(email, password)
      setStep('confirm')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Sign up failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleConfirm = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await doConfirmSignUp(email, code)
      onSuccess()
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Confirmation failed.')
    } finally {
      setLoading(false)
    }
  }

  if (step === 'confirm') {
    return (
      <AuthCard title="Check your email" subtitle={`We sent a 6-digit code to ${email}`}>
        <form onSubmit={handleConfirm} className="flex flex-col gap-4" noValidate>
          <Input
            label="Confirmation code"
            type="text"
            inputMode="numeric"
            autoComplete="one-time-code"
            value={code}
            onChange={e => setCode(e.target.value)}
            required
          />
          {error && (
            <p className="text-sm text-red-600 dark:text-red-400 rounded-lg bg-red-50 dark:bg-red-900/20 px-3 py-2" role="alert">
              {error}
            </p>
          )}
          <Button type="submit" loading={loading} className="w-full mt-1">Confirm account</Button>
        </form>
        <p className="mt-5 text-center text-sm text-[var(--color-text-muted)]">
          Wrong email?{' '}
          <button onClick={() => setStep('register')} className="text-primary-600 dark:text-primary-400 font-medium hover:underline">
            Go back
          </button>
        </p>
      </AuthCard>
    )
  }

  return (
    <AuthCard title="Create account" subtitle="Join AmanAI to access NUST Bank AI assistant">
      <form onSubmit={handleRegister} className="flex flex-col gap-4" noValidate>
        <Input label="Email" type="email" autoComplete="email" value={email} onChange={e => setEmail(e.target.value)} required />
        <Input
          label="Password"
          type="password"
          autoComplete="new-password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          hint="Minimum 8 characters"
          required
        />
        {error && (
          <p className="text-sm text-red-600 dark:text-red-400 rounded-lg bg-red-50 dark:bg-red-900/20 px-3 py-2" role="alert">
            {error}
          </p>
        )}
        <Button type="submit" loading={loading} className="w-full mt-1">Create account</Button>
      </form>
      <p className="mt-5 text-center text-sm text-[var(--color-text-muted)]">
        Already have an account?{' '}
        <button onClick={onSignIn} className="text-primary-600 dark:text-primary-400 font-medium hover:underline">Sign in</button>
      </p>
    </AuthCard>
  )
}
