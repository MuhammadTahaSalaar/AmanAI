import { Card } from '@/components/ui/Card'

interface AuthCardProps {
  title:    string
  subtitle?: string
  children: React.ReactNode
}

export function AuthCard({ title, subtitle, children }: AuthCardProps) {
  return (
    <div className="min-h-screen bg-[var(--color-surface)] flex flex-col items-center justify-center px-4 py-12">
      {/* Brand mark */}
      <div className="flex items-center gap-2.5 mb-8">
        <div className="w-9 h-9 rounded-xl bg-primary-600 flex items-center justify-center shadow-glow">
          <span className="text-white text-base font-bold select-none">A</span>
        </div>
        <div>
          <div className="font-bold text-[var(--color-text)] leading-none">AmanAI</div>
          <div className="text-xs text-[var(--color-text-muted)]">NUST Bank</div>
        </div>
      </div>

      <Card className="w-full max-w-sm">
        <div className="mb-6">
          <h1 className="text-xl font-semibold text-[var(--color-text)]">{title}</h1>
          {subtitle && (
            <p className="text-sm text-[var(--color-text-muted)] mt-1">{subtitle}</p>
          )}
        </div>
        {children}
      </Card>
    </div>
  )
}
