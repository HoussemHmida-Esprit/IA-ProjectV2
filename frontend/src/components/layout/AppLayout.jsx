import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Target, 
  Brain, 
  Menu, 
  X,
  Activity,
  Database,
  Calendar
} from 'lucide-react'
import StatusIndicator from '../ui/StatusIndicator'
import axios from 'axios'

const navigation = [
  { name: 'Overview Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Live Prediction', href: '/predict', icon: Target },
  { name: 'Model Explainability', href: '/xai', icon: Brain },
  { name: 'Forecasting', href: '/forecast', icon: Calendar },
]

export default function AppLayout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [apiStatus, setApiStatus] = useState('offline')
  const [modelsStatus, setModelsStatus] = useState('offline')
  const location = useLocation()

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 30000) // Check every 30s
    return () => clearInterval(interval)
  }, [])

  const checkHealth = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/health')
      setApiStatus(response.data.status === 'online' ? 'online' : 'offline')
      setModelsStatus(response.data.models_loaded ? 'synced' : 'offline')
    } catch (error) {
      setApiStatus('offline')
      setModelsStatus('offline')
    }
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 bg-slate-850 transition-all duration-300 ${
          sidebarOpen ? 'w-64' : 'w-20'
        }`}
      >
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center justify-between px-6 border-b border-slate-700">
            {sidebarOpen && (
              <h1 className="text-xl font-bold text-white">AccidentAI</h1>
            )}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-slate-400 hover:text-white transition-colors"
            >
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-1 px-3 py-4">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href
              const Icon = item.icon
              
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-primary-600 text-white'
                      : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  <Icon size={20} />
                  {sidebarOpen && (
                    <span className="font-medium">{item.name}</span>
                  )}
                </Link>
              )
            })}
          </nav>
        </div>
      </aside>

      {/* Main Content */}
      <div
        className={`transition-all duration-300 ${
          sidebarOpen ? 'ml-64' : 'ml-20'
        }`}
      >
        {/* Header */}
        <header className="sticky top-0 z-40 bg-white border-b border-slate-200">
          <div className="flex h-16 items-center justify-between px-8">
            <h2 className="text-lg font-semibold text-slate-900">
              {navigation.find(item => item.href === location.pathname)?.name || 'Dashboard'}
            </h2>
            
            <div className="flex items-center gap-6">
              <StatusIndicator
                icon={Activity}
                label="API"
                status={apiStatus}
              />
              <StatusIndicator
                icon={Database}
                label="Models"
                status={modelsStatus}
              />
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="p-8">
          {children}
        </main>
      </div>
    </div>
  )
}
