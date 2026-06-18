'use client'

import { useRef, useState, useCallback } from 'react'
import { SendHorizonal } from 'lucide-react'
import clsx from 'clsx'

const MAX_CHARS = 2000

interface ChatInputProps {
  onSend:   (message: string) => void
  disabled: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue]     = useState('')
  const textareaRef           = useRef<HTMLTextAreaElement>(null)

  const handleSend = useCallback(() => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }, [value, disabled, onSend])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const next = e.target.value
    if (next.length > MAX_CHARS) return
    setValue(next)
    // Auto-grow
    e.target.style.height = 'auto'
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`
  }

  const remaining  = MAX_CHARS - value.length
  const nearLimit  = remaining < 200

  return (
    <div className="border-t border-[var(--color-border)] bg-[var(--color-panel)] px-4 py-3">
      <div className="max-w-3xl mx-auto">
        <div className={clsx(
          'flex items-end gap-2 rounded-2xl border px-3 py-2 transition-colors',
          'bg-[var(--color-surface)] focus-within:border-primary-500 focus-within:ring-1 focus-within:ring-primary-500',
          disabled ? 'border-[var(--color-border)] opacity-60' : 'border-[var(--color-border)]',
        )}>
          <textarea
            ref={textareaRef}
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder="Ask about NUST Bank products, rates, or services…"
            rows={1}
            aria-label="Chat message input"
            className={clsx(
              'flex-1 resize-none bg-transparent text-sm text-[var(--color-text)]',
              'placeholder:text-[var(--color-text-muted)] focus:outline-none',
              'min-h-[36px] max-h-[160px] py-1.5 leading-relaxed',
            )}
          />
          <button
            onClick={handleSend}
            disabled={disabled || !value.trim()}
            aria-label="Send message"
            className={clsx(
              'shrink-0 w-9 h-9 flex items-center justify-center rounded-xl transition-all duration-150',
              value.trim() && !disabled
                ? 'bg-primary-600 text-white hover:bg-primary-700 shadow-sm'
                : 'bg-[var(--color-border)] text-[var(--color-text-muted)] cursor-not-allowed',
            )}
          >
            <SendHorizonal size={16} />
          </button>
        </div>
        {nearLimit && (
          <p className="mt-1 text-right text-xs text-[var(--color-text-muted)]">
            {remaining} characters remaining
          </p>
        )}
        <p className="mt-1 text-center text-xs text-[var(--color-text-muted)]">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
