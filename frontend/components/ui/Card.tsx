import clsx from 'clsx'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  padded?: boolean
}

export function Card({ padded = true, className, children, ...props }: CardProps) {
  return (
    <div
      className={clsx(
        'bg-[var(--color-panel)] border border-[var(--color-border)] rounded-2xl shadow-card',
        padded && 'p-6',
        className,
      )}
      {...props}
    >
      {children}
    </div>
  )
}
