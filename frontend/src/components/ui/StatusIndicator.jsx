export default function StatusIndicator({ icon: Icon, label, status }) {
  const statusConfig = {
    online: { color: 'text-green-600', bg: 'bg-green-100', dot: 'bg-green-500' },
    synced: { color: 'text-blue-600', bg: 'bg-blue-100', dot: 'bg-blue-500' },
    offline: { color: 'text-red-600', bg: 'bg-red-100', dot: 'bg-red-500' },
  }

  const config = statusConfig[status] || statusConfig.offline

  return (
    <div className="flex items-center gap-2">
      <div className={`p-1.5 rounded-lg ${config.bg}`}>
        <Icon size={16} className={config.color} />
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-sm font-medium text-slate-700">{label}:</span>
        <div className="flex items-center gap-1">
          <div className={`w-1.5 h-1.5 rounded-full ${config.dot}`} />
          <span className={`text-sm font-medium capitalize ${config.color}`}>
            {status}
          </span>
        </div>
      </div>
    </div>
  )
}
