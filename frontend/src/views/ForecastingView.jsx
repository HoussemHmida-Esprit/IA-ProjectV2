import { useState } from 'react'
import { Calendar, TrendingUp, BarChart3 } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import Card from '../components/ui/Card'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import axios from 'axios'

export default function ForecastingView() {
  const [forecastDays, setForecastDays] = useState(7)
  const [forecast, setForecast] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const generateForecast = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await axios.post('http://localhost:8000/api/forecast', {
        days: forecastDays
      })
      setForecast(response.data)
    } catch (err) {
      console.error('Forecast failed:', err)
      setError(err.response?.data?.detail || 'Forecast failed. Make sure the LSTM model is trained.')
    } finally {
      setLoading(false)
    }
  }

  const chartData = forecast ? forecast.dates.map((date, idx) => ({
    date: new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    predicted: forecast.predictions[idx]
  })) : []

  const maxPrediction = forecast ? Math.max(...forecast.predictions) : 0
  const peakDay = forecast ? forecast.dates[forecast.predictions.indexOf(maxPrediction)] : null

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Control Panel */}
      <Card title="LSTM Time-Series Forecasting" subtitle="Predict future accident counts">
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Forecast Horizon (days)
            </label>
            <input
              type="range"
              min={1}
              max={30}
              value={forecastDays}
              onChange={(e) => setForecastDays(parseInt(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
            />
            <div className="flex justify-between text-sm text-slate-500 mt-1">
              <span>1 day</span>
              <span className="font-medium text-slate-900">{forecastDays} days</span>
              <span>30 days</span>
            </div>
          </div>
          
          <button
            onClick={generateForecast}
            disabled={loading}
            className="btn-primary flex items-center gap-2 mt-6"
          >
            {loading ? (
              <>
                <LoadingSpinner size="sm" />
                <span>Forecasting...</span>
              </>
            ) : (
              <>
                <Calendar size={18} />
                <span>Generate Forecast</span>
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800">
            {error}
          </div>
        )}
      </Card>

      {forecast && (
        <>
          {/* Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600 mb-1">Total Predicted</p>
                  <p className="text-2xl font-bold text-slate-900">{forecast.total.toLocaleString()}</p>
                </div>
                <div className="p-3 rounded-lg bg-blue-100">
                  <BarChart3 size={24} className="text-blue-600" />
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600 mb-1">Average per Day</p>
                  <p className="text-2xl font-bold text-slate-900">{forecast.average.toFixed(1)}</p>
                </div>
                <div className="p-3 rounded-lg bg-green-100">
                  <TrendingUp size={24} className="text-green-600" />
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600 mb-1">Peak Day</p>
                  <p className="text-2xl font-bold text-slate-900">{maxPrediction}</p>
                  <p className="text-xs text-slate-500 mt-1">
                    {peakDay && new Date(peakDay).toLocaleDateString('en-US', { 
                      month: 'short', 
                      day: 'numeric' 
                    })}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-orange-100">
                  <Calendar size={24} className="text-orange-600" />
                </div>
              </div>
            </Card>
          </div>

          {/* Chart */}
          <Card title="Forecast Visualization" subtitle={`Next ${forecastDays} days`}>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#64748b"
                    style={{ fontSize: '12px' }}
                  />
                  <YAxis 
                    stroke="#64748b"
                    style={{ fontSize: '12px' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'white', 
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="predicted" 
                    stroke="#ef4444" 
                    strokeWidth={3}
                    dot={{ fill: '#ef4444', r: 5 }}
                    activeDot={{ r: 7 }}
                    name="Predicted Accidents"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* Forecast Table */}
          <Card title="Daily Forecast" subtitle="Detailed predictions">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-200">
                    <th className="text-left py-3 px-4 text-sm font-semibold text-slate-700">Date</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-slate-700">Day</th>
                    <th className="text-right py-3 px-4 text-sm font-semibold text-slate-700">Predicted Accidents</th>
                  </tr>
                </thead>
                <tbody>
                  {forecast.dates.map((date, idx) => (
                    <tr key={date} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="py-3 px-4 text-sm text-slate-900">
                        {new Date(date).toLocaleDateString('en-US', { 
                          year: 'numeric',
                          month: 'short', 
                          day: 'numeric' 
                        })}
                      </td>
                      <td className="py-3 px-4 text-sm text-slate-600">
                        {new Date(date).toLocaleDateString('en-US', { weekday: 'long' })}
                      </td>
                      <td className="py-3 px-4 text-sm font-semibold text-slate-900 text-right">
                        {forecast.predictions[idx]}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* Information */}
      <Card title="About LSTM Forecasting">
        <div className="prose prose-sm max-w-none text-slate-600">
          <p>
            This forecasting system uses a Long Short-Term Memory (LSTM) neural network to predict 
            future accident counts based on historical patterns.
          </p>
          
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="font-semibold text-blue-900 mb-2">How It Works</p>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Analyzes last 30 days of accident data</li>
                <li>• Learns temporal patterns and trends</li>
                <li>• Predicts future counts iteratively</li>
                <li>• Updates predictions with each forecast</li>
              </ul>
            </div>
            
            <div className="p-4 bg-amber-50 rounded-lg">
              <p className="font-semibold text-amber-900 mb-2">Limitations</p>
              <ul className="text-sm text-amber-800 space-y-1">
                <li>• Assumes patterns continue</li>
                <li>• Cannot predict unprecedented events</li>
                <li>• Accuracy decreases for longer horizons</li>
                <li>• Weather/events not included</li>
              </ul>
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}
