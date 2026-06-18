'use client'

import { Moon, Sun, LogOut, Upload } from 'lucide-react'
import { useTheme } from '@/components/providers/ThemeProvider'
import { Button }   from '@/components/ui/Button'
import type { AuthUser } from '@/lib/auth'

interface HeaderProps {
  user:          AuthUser | null
  onSignOut:     () => void
  activeTab:     'chat' | 'knowledge'
  onTabChange:   (tab: 'chat' | 'knowledge') => void
}

export function Header({ user, onSignOut, activeTab, onTabChange }: HeaderProps) {
  const { theme, toggle } = useTheme()

  return (
    <header className="border-b border-[var(--color-border)] bg-[var(--color-panel)] px-4 h-14 flex items-center gap-4 shrink-0">
      {/* Brand */}
      <div className="flex items-center gap-2.5 mr-2">
        <div className="w-7 h-7 rounded-lg bg-primary-600 flex items-center justify-center">
          <span className="text-white text-xs font-bold select-none">A</span>
        </div>
        <span className="font-semibold text-[var(--color-text)] tracking-tight">AmanAI</span>
        <span className="hidden sm:inline text-xs text-[var(--color-text-muted)] border border-[var(--color-border)] px-2 py-0.5 rounded-full">
          NUST Bank
        </span>
      </div>

      {/* Tabs */}
      <nav className="flex items-center gap-1" aria-label="Main navigation">
        <button
          onClick={() => onTabChange('chat')}
          aria-current={activeTab === 'chat' ? 'page' : undefined}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            activeTab === 'chat'
              ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
              : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-black/5 dark:hover:bg-white/5'
          }`}
        >
          Chat
        </button>

        {user?.isAdmin && (
          <button
            onClick={() => onTabChange('knowledge')}
            aria-current={activeTab === 'knowledge' ? 'page' : undefined}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'knowledge'
                ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-black/5 dark:hover:bg-white/5'
            }`}
          >
            <Upload size={13} />
            Knowledge Base
          </button>
        )}
      </nav>

      {/* Right controls */}
      <div className="ml-auto flex items-center gap-2">
        {user && (
          <span className="hidden sm:inline text-xs text-[var(--color-text-muted)] truncate max-w-[140px]">
            {user.email}
          </span>
        )}
        <Button variant="ghost" size="sm" onClick={toggle} aria-label="Toggle theme">
          {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
        </Button>
        {user && (
          <Button variant="ghost" size="sm" onClick={onSignOut} aria-label="Sign out">
            <LogOut size={15} />
            <span className="hidden sm:inline">Sign out</span>
          </Button>
        )}
      </div>
    </header>
  )
}
