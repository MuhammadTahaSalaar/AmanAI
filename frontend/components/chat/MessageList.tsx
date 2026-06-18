'use client'

import { useEffect, useRef } from 'react'
import { MessageBubble, type Message } from './MessageBubble'
import { TypingIndicator }             from './TypingIndicator'
import { EmptyState }                  from './EmptyState'

interface MessageListProps {
  messages:  Message[]
  loading:   boolean
  onExample: (q: string) => void
}

export function MessageList({ messages, loading, onExample }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, loading])

  return (
    <div
      className="flex-1 overflow-y-auto px-4 py-6"
      role="log"
      aria-live="polite"
      aria-label="Conversation"
    >
      <div className="max-w-3xl mx-auto flex flex-col gap-5">
        {messages.length === 0 && !loading ? (
          <EmptyState onExample={onExample} />
        ) : (
          messages.map(msg => <MessageBubble key={msg.id} message={msg} />)
        )}
        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
