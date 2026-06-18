import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AlertTriangle } from 'lucide-react'
import { CitationsPanel } from './CitationsPanel'
import type { Citation } from '@/lib/api'
import clsx from 'clsx'

export type Message = {
  id:        string
  role:      'user' | 'assistant'
  content:   string
  citations?: Citation[]
  refused?:  boolean
  pending?:  boolean
}

interface MessageBubbleProps {
  message: Message
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end animate-slide-up">
        <div className="max-w-[75%] bg-primary-600 text-white px-4 py-3 rounded-2xl rounded-br-sm shadow-sm text-sm leading-relaxed">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-end gap-3 animate-slide-up">
      <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/40 flex items-center justify-center shrink-0">
        <span className="text-primary-700 dark:text-primary-300 text-xs font-bold select-none">AI</span>
      </div>

      <div className="max-w-[80%] flex flex-col gap-2">
        {message.refused ? (
          <div className="rounded-2xl rounded-bl-sm border border-[var(--color-warning-border)] bg-[var(--color-warning-bg)] px-4 py-3 shadow-card">
            <div className="flex items-center gap-2 mb-2 text-[var(--color-warning-text)]">
              <AlertTriangle size={15} />
              <span className="text-xs font-semibold uppercase tracking-wide">Out of scope</span>
            </div>
            <p className="text-sm text-[var(--color-warning-text)] leading-relaxed">
              {message.content}
            </p>
          </div>
        ) : (
          <div
            className={clsx(
              'bg-[var(--color-panel)] border border-[var(--color-border)] rounded-2xl rounded-bl-sm px-4 py-3 shadow-card',
              message.pending && 'opacity-60',
            )}
          >
            <div className="prose-chat">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {message.citations && message.citations.length > 0 && (
          <CitationsPanel citations={message.citations} />
        )}
      </div>
    </div>
  )
}
