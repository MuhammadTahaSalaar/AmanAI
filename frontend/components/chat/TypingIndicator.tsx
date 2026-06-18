export function TypingIndicator() {
  return (
    <div className="flex items-end gap-3 animate-fade-in" role="status" aria-label="Assistant is typing">
      <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/40 flex items-center justify-center shrink-0">
        <span className="text-primary-600 dark:text-primary-400 text-xs font-bold">AI</span>
      </div>
      <div className="bg-[var(--color-panel)] border border-[var(--color-border)] rounded-2xl rounded-bl-sm px-4 py-3 shadow-card">
        <div className="flex items-center gap-1.5 h-5">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
        </div>
      </div>
    </div>
  )
}
