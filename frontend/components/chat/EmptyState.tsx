import { Sparkles } from 'lucide-react'

const EXAMPLE_QUESTIONS = [
  'What savings account options does NUST Bank offer?',
  'What are the current fixed deposit rates?',
  'How do I apply for a home loan?',
  'What is the minimum balance for a current account?',
]

interface EmptyStateProps {
  onExample: (question: string) => void
}

export function EmptyState({ onExample }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-8 px-4 py-12 text-center animate-fade-in">
      <div className="flex flex-col items-center gap-3">
        <div className="w-16 h-16 rounded-2xl bg-primary-50 dark:bg-primary-900/30 flex items-center justify-center">
          <Sparkles className="text-primary-600 dark:text-primary-400" size={28} />
        </div>
        <h2 className="text-xl font-semibold text-[var(--color-text)]">
          How can I help you today?
        </h2>
        <p className="text-sm text-[var(--color-text-muted)] max-w-sm">
          Ask me anything about NUST Bank products, rates, and services. I&apos;ll answer using verified sources.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full max-w-xl">
        {EXAMPLE_QUESTIONS.map(q => (
          <button
            key={q}
            onClick={() => onExample(q)}
            className="text-left px-4 py-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] hover:border-primary-400 hover:bg-primary-50 dark:hover:bg-primary-900/20 transition-all text-sm text-[var(--color-text)] shadow-card hover:shadow-card-md"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}
