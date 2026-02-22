export default function LoadingSpinner({ size = 'md' }) {
  const sizeClasses = {
    sm: 'w-4 h-4 border-2',
    md: 'w-6 h-6 border-2',
    lg: 'w-8 h-8 border-3',
  }

  return (
    <div
      className={`${sizeClasses[size]} border-primary-200 border-t-primary-600 rounded-full animate-spin`}
    />
  )
}
