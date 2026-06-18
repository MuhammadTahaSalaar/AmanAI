'use client'

import { useState } from 'react'
import { ChevronDown, ChevronUp, BookOpen } from 'lucide-react'
import type { Citation } from '@/lib/api'
import clsx from 'clsx'

interface CitationsPanelProps {
  citations: Citation[]
}

export function CitationsPanel({ citations }: CitationsPanelProps) {
  const [open, setOpen] = useState(false)
  if (citations.length === 0) return null

  return (
    <div className="mt-2 border border-[var(--color-border)] rounded-xl overflow-hidden text-sm">
      <button
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2 text-[var(--color-text-muted)] hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
      >
        <BookOpen size={14} />
        <span className="font-medium">{citations.length} source{citations.length !== 1 ? 's' : ''}</span>
        <span className="ml-auto">{open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</span>
      </button>
      {open && (
        <ul className="divide-y divide-[var(--color-border)] animate-fade-in">
          {citations.map((c, i) => (
            <li key={i} className="px-3 py-2.5 bg-[var(--color-surface)]">
              {(c.product || c.source) && (
                <div className="flex gap-2 mb-1 flex-wrap">
                  {c.product && (
                    <span className="inline-block text-xs font-semibold text-primary-700 dark:text-primary-300 bg-primary-50 dark:bg-primary-900/30 px-2 py-0.5 rounded-full">
                      {c.product}
                    </span>
                  )}
                  {c.source && (
                    <span className="inline-block text-xs text-[var(--color-text-muted)]">
                      {c.source}
                    </span>
                  )}
                </div>
              )}
              <p className={clsx('text-xs text-[var(--color-text-muted)] leading-relaxed line-clamp-3')}>
                {c.content}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
