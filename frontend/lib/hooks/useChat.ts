'use client'

import { useState, useCallback } from 'react'
import { postChat, type ChatMessage } from '@/lib/api'
import type { Message } from '@/components/chat/MessageBubble'

let idCounter = 0
const nextId = () => `msg-${++idCounter}`

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState<string | null>(null)

  const send = useCallback(async (text: string) => {
    setError(null)

    // Optimistic user bubble
    const userMsg: Message = { id: nextId(), role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    // Build history from current messages (exclude the one we just added)
    const history: ChatMessage[] = messages.map(m => ({
      role:    m.role,
      content: m.content,
    }))

    try {
      const res = await postChat(text, history)
      const assistantMsg: Message = {
        id:        nextId(),
        role:      'assistant',
        content:   res.answer,
        citations: res.citations,
        refused:   res.refused,
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Something went wrong.'
      setError(msg)
      // Remove the optimistic user message on failure
      setMessages(prev => prev.filter(m => m.id !== userMsg.id))
    } finally {
      setLoading(false)
    }
  }, [messages])

  return { messages, loading, error, send }
}
