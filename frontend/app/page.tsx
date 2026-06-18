'use client'

import { useState, useCallback } from 'react'
import { useAuth }        from '@/lib/hooks/useAuth'
import { useChat }        from '@/lib/hooks/useChat'
import { Header }         from '@/components/layout/Header'
import { MessageList }    from '@/components/chat/MessageList'
import { ChatInput }      from '@/components/chat/ChatInput'
import { UploadPanel }    from '@/components/knowledge/UploadPanel'
import { SignInForm }     from '@/components/auth/SignInForm'
import { SignUpForm }     from '@/components/auth/SignUpForm'
import { AlertCircle }    from 'lucide-react'

type AuthScreen = 'signin' | 'signup'
type AppTab     = 'chat' | 'knowledge'

export default function HomePage() {
  const { user, loading: authLoading, signOut, refresh } = useAuth()
  const { messages, loading: chatLoading, error, send }  = useChat()

  const [authScreen, setAuthScreen] = useState<AuthScreen>('signin')
  const [activeTab,  setActiveTab]  = useState<AppTab>('chat')

  const handleSend = useCallback((text: string) => {
    send(text)
  }, [send])

  // ── Loading splash ──────────────────────────────────────────────────────────
  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--color-surface)]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary-600 flex items-center justify-center shadow-glow animate-pulse-slow">
            <span className="text-white text-sm font-bold">A</span>
          </div>
          <p className="text-sm text-[var(--color-text-muted)]">Loading…</p>
        </div>
      </div>
    )
  }

  // ── Auth screens ────────────────────────────────────────────────────────────
  if (!user) {
    if (authScreen === 'signup') {
      return (
        <SignUpForm
          onSuccess={() => { refresh(); setAuthScreen('signin') }}
          onSignIn={() => setAuthScreen('signin')}
        />
      )
    }
    return (
      <SignInForm
        onSuccess={refresh}
        onSignUp={() => setAuthScreen('signup')}
      />
    )
  }

  // ── Authenticated app ───────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[var(--color-surface)]">
      <Header
        user={user}
        onSignOut={signOut}
        activeTab={activeTab}
        onTabChange={tab => {
          // Guard: only admins can access knowledge tab
          if (tab === 'knowledge' && !user.isAdmin) return
          setActiveTab(tab)
        }}
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        {activeTab === 'knowledge' && user.isAdmin ? (
          <div className="flex-1 overflow-y-auto">
            <UploadPanel />
          </div>
        ) : (
          <>
            {error && (
              <div className="flex items-center gap-2 px-4 py-2 bg-red-50 dark:bg-red-900/20 border-b border-red-200 dark:border-red-800 text-sm text-red-700 dark:text-red-300" role="alert">
                <AlertCircle size={14} className="shrink-0" />
                {error}
              </div>
            )}
            <MessageList
              messages={messages}
              loading={chatLoading}
              onExample={handleSend}
            />
            <ChatInput onSend={handleSend} disabled={chatLoading} />
          </>
        )}
      </main>
    </div>
  )
}
