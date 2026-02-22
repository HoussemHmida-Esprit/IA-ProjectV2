export default function Card({ title, subtitle, children, className = '' }) {
  return (
    <div className={`card p-6 ${className}`}>
      {(title || subtitle) && (
        <div className="mb-6">
          {title && (
            <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
          )}
          {subtitle && (
            <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
          )}
        </div>
      )}
      {children}
    </div>
  )
}
