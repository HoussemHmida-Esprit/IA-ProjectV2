import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ScatterChart, Scatter, ZAxis } from 'recharts'
import { Brain, TrendingUp, TrendingDown, RefreshCw, Info } from 'lucide-react'
import Card from '../components/ui/Card'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import axios from 'axios'

const FEATURE_DESCRIPTIONS = {
  'Lighting': 'Lighting conditions at time of accident',
  'Weather': 'Atmospheric conditions',
  'Location': 'Urban vs rural area',
  'Intersection': 'Type of intersection',
  'Hour': 'Time of day (0-23)',
  'Day of Week': 'Day of the week',
  'Month': 'Month of the year',
  'People Involved': 'Number of people in accident',
  'Light Injuries': 'Number of light injuries'
}

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-white p-4 rounded-lg shadow-lg border border-slate-200">
        <p className="font-semibold text-slate-900 mb-1">{data.feature}</p>
        <p className={`text-sm font-medium ${data.value > 0 ? 'text-red-600' : 'text-blue-600'}`}>
          Impact: {data.display || data.value.toFixed(3)}
        </p>
        {data.inputValue !== undefined && (
          <p className="text-xs text-slate-600 mt-1">
            Input value: {data.inputValue}
          </p>
        )}
        <p className="text-xs text-slate-500 mt-2">
          {FEATURE_DESCRIPTIONS[data.feature] || 'Feature impact on prediction'}
        </p>
      </div>
    )
  }
  return null
}

export default function XaiDashboard() {
  const [shapData, setShapData] = useState([])
  const [globalImportance, setGlobalImportance] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [viewMode, setViewMode] = useState('global') // 'global' or 'specific'
  const [predictionParams, setPredictionParams] = useState({
    lighting: 1,
    location: 1,
    intersection: 1,
    day_of_week: 0,
    hour: 12,
    num_users: 2
  })

  useEffect(() => {
    loadGlobalImportance()
  }, [])

  const loadGlobalImportance = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.get('http://localhost:8000/api/shap/importance')
      const importanceData = response.data.features.map((feature, idx) => ({
        feature: feature,
        value: response.data.importance[idx],
        display: response.data.importance[idx].toFixed(3)
      }))
      setGlobalImportance(importanceData)
      setViewMode('global')
      setLoading(false)
    } catch (err) {
      console.error('Failed to load SHAP importance:', err)
      setError('Failed to load feature importance. Make sure the backend is running.')
      setLoading(false)
    }
  }

  const loadShapForPrediction = async () => {
    setLoading(true)
    setError(null)
    try {
      console.log('Sending SHAP request with params:', predictionParams)
      const response = await axios.post('http://localhost:8000/api/shap', predictionParams)
      console.log('SHAP response:', response.data)
      
      const shapValues = response.data.feature_details.map(item => ({
        feature: item.feature,
        value: item.value,
        display: item.value >= 0 ? `+${item.value.toFixed(3)}` : item.value.toFixed(3),
        inputValue: item.input_value
      }))
      
      console.log('Processed SHAP values:', shapValues)
      setShapData(shapValues)
      setViewMode('specific')
      setLoading(false)
    } catch (err) {
      console.error('Failed to load SHAP values:', err)
      console.error('Error response:', err.response?.data)
      setError(
        err.response?.data?.detail || 
        'Failed to calculate SHAP values. Check browser console for details.'
      )
      setLoading(false)
    }
  }

  const displayData = viewMode === 'specific' && shapData.length > 0 ? shapData : globalImportance || []
  const sortedData = [...displayData].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
  const topPositive = sortedData.find(d => d.value > 0)
  const topNegative = sortedData.reverse().find(d => d.value < 0)

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Control Panel */}
      <Card title="SHAP Analysis Control" subtitle="Understand model predictions">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={loadGlobalImportance}
                disabled={loading}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  viewMode === 'global'
                    ? 'bg-primary-600 text-white'
                    : 'border border-slate-300 hover:bg-slate-50'
                }`}
              >
                Global Importance
              </button>
              <button
                onClick={loadShapForPrediction}
                disabled={loading}
                className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
                  viewMode === 'specific'
                    ? 'bg-primary-600 text-white'
                    : 'border border-slate-300 hover:bg-slate-50'
                }`}
              >
                {loading && viewMode === 'specific' ? (
                  <>
                    <LoadingSpinner size="sm" />
                    <span>Calculating...</span>
                  </>
                ) : (
                  <>
                    <RefreshCw size={16} />
                    <span>Specific Prediction</span>
                  </>
                )}
              </button>
            </div>
            <div className="text-sm text-slate-600">
              {viewMode === 'specific' ? 'Showing: Prediction-Specific SHAP' : 'Showing: Global Feature Importance'}
            </div>
          </div>

          {viewMode === 'specific' && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-900 mb-3 font-medium">Current Parameters:</p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div>
                  <label className="text-xs text-blue-700 block mb-1">Lighting</label>
                  <input
                    type="number"
                    min="1"
                    max="5"
                    value={predictionParams.lighting}
                    onChange={(e) => setPredictionParams({...predictionParams, lighting: parseInt(e.target.value)})}
                    className="w-full px-2 py-1 text-sm border border-blue-300 rounded"
                  />
                </div>
                <div>
                  <label className="text-xs text-blue-700 block mb-1">Location</label>
                  <input
                    type="number"
                    min="1"
                    max="2"
                    value={predictionParams.location}
                    onChange={(e) => setPredictionParams({...predictionParams, location: parseInt(e.target.value)})}
                    className="w-full px-2 py-1 text-sm border border-blue-300 rounded"
                  />
                </div>
                <div>
                  <label className="text-xs text-blue-700 block mb-1">Intersection</label>
                  <input
                    type="number"
                    min="1"
                    max="9"
                    value={predictionParams.intersection}
                    onChange={(e) => setPredictionParams({...predictionParams, intersection: parseInt(e.target.value)})}
                    className="w-full px-2 py-1 text-sm border border-blue-300 rounded"
                  />
                </div>
                <div>
                  <label className="text-xs text-blue-700 block mb-1">Day (0-6)</label>
                  <input
                    type="number"
                    min="0"
                    max="6"
                    value={predictionParams.day_of_week}
                    onChange={(e) => setPredictionParams({...predictionParams, day_of_week: parseInt(e.target.value)})}
                    className="w-full px-2 py-1 text-sm border border-blue-300 rounded"
                  />
                </div>
                <div>
                  <label className="text-xs text-blue-700 block mb-1">Hour (0-23)</label>
                  <input
                    type="number"
                    min="0"
                    max="23"
                    value={predictionParams.hour}
                    onChange={(e) => setPredictionParams({...predictionParams, hour: parseInt(e.target.value)})}
                    className="w-full px-2 py-1 text-sm border border-blue-300 rounded"
                  />
                </div>
                <div>
                  <label className="text-xs text-blue-700 block mb-1">People</label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={predictionParams.num_users}
                    onChange={(e) => setPredictionParams({...predictionParams, num_users: parseInt(e.target.value)})}
                    className="w-full px-2 py-1 text-sm border border-blue-300 rounded"
                  />
                </div>
              </div>
              <p className="text-xs text-blue-700 mt-2">
                Adjust parameters above and click "Specific Prediction" to recalculate SHAP values
              </p>
            </div>
          )}
          
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800">
              {error}
            </div>
          )}
        </div>
      </Card>

      {/* Header Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Brain size={24} className="text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-slate-600">Analysis Type</p>
              <p className="font-semibold text-slate-900">
                {viewMode === 'global' ? 'Global Importance' : 'Prediction-Specific'}
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-red-100 rounded-lg">
              <TrendingUp size={24} className="text-red-600" />
            </div>
            <div>
              <p className="text-sm text-slate-600">Top Risk Factor</p>
              <p className="font-semibold text-slate-900">{topPositive?.feature || 'N/A'}</p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <TrendingDown size={24} className="text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-slate-600">Top Protective Factor</p>
              <p className="font-semibold text-slate-900">{topNegative?.feature || 'N/A'}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* SHAP Force Plot */}
      <Card 
        title="SHAP Feature Importance" 
        subtitle={viewMode === 'global' ? 'Overall feature importance across all predictions' : 'Feature impact on current prediction'}
      >
        {loading ? (
          <div className="h-96 flex items-center justify-center">
            <LoadingSpinner size="lg" />
          </div>
        ) : displayData.length === 0 ? (
          <div className="h-96 flex items-center justify-center text-slate-500">
            No data available. Click a button above to load SHAP analysis.
          </div>
        ) : (
          <>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={displayData}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 140, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis 
                    type="number" 
                    stroke="#64748b"
                    style={{ fontSize: '12px' }}
                  />
                  <YAxis 
                    type="category" 
                    dataKey="feature" 
                    stroke="#64748b"
                    style={{ fontSize: '13px', fontWeight: 500 }}
                    width={130}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {displayData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.value > 0 ? '#dc2626' : '#2563eb'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Legend */}
            <div className="flex items-center justify-center gap-6 mt-6 pt-6 border-t border-slate-200">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-600 rounded" />
                <span className="text-sm text-slate-600">Increases Risk</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-blue-600 rounded" />
                <span className="text-sm text-slate-600">Decreases Risk</span>
              </div>
            </div>
          </>
        )}
      </Card>

      {/* Feature Importance Table */}
      <Card title="Feature Importance Rankings" subtitle="Detailed breakdown of feature impacts">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-3 px-4 text-sm font-semibold text-slate-700">Rank</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-slate-700">Feature</th>
                <th className="text-right py-3 px-4 text-sm font-semibold text-slate-700">Impact</th>
                <th className="text-center py-3 px-4 text-sm font-semibold text-slate-700">Direction</th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-slate-700">Description</th>
              </tr>
            </thead>
            <tbody>
              {sortedData.slice(0, 10).map((item, idx) => (
                <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                  <td className="py-3 px-4 text-sm font-medium text-slate-900">{idx + 1}</td>
                  <td className="py-3 px-4 text-sm font-medium text-slate-900">{item.feature}</td>
                  <td className="py-3 px-4 text-sm font-semibold text-right">
                    {Math.abs(item.value).toFixed(3)}
                  </td>
                  <td className="py-3 px-4 text-center">
                    {item.value > 0 ? (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                        <TrendingUp size={12} className="mr-1" />
                        Risk
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        <TrendingDown size={12} className="mr-1" />
                        Protective
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-xs text-slate-600">
                    {FEATURE_DESCRIPTIONS[item.feature] || 'Feature impact'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Interpretation Summary */}
      <Card title="Model Interpretation" subtitle="Plain-language explanation">
        <div className="space-y-4">
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm font-semibold text-red-900 mb-2 flex items-center gap-2">
              <TrendingUp size={16} />
              Primary Risk Factors
            </p>
            <p className="text-sm text-red-800">
              {viewMode === 'specific' ? (
                <>
                  The model predicted <span className="font-semibold">higher severity</span> primarily 
                  because of <span className="font-semibold">{topPositive?.feature}</span> (impact: {topPositive?.display}). 
                  This feature significantly increases the predicted accident severity.
                </>
              ) : (
                <>
                  Across all predictions, <span className="font-semibold">{topPositive?.feature}</span> is 
                  the most important feature for determining accident severity. Features with high importance 
                  have the strongest influence on model predictions.
                </>
              )}
            </p>
          </div>

          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm font-semibold text-blue-900 mb-2 flex items-center gap-2">
              <TrendingDown size={16} />
              Mitigating Factors
            </p>
            <p className="text-sm text-blue-800">
              {viewMode === 'specific' ? (
                <>
                  The severity was reduced by <span className="font-semibold">{topNegative?.feature}</span> (impact: {topNegative?.display}). 
                  These protective factors helped lower the overall risk assessment.
                </>
              ) : (
                <>
                  <span className="font-semibold">{topNegative?.feature}</span> tends to reduce predicted 
                  severity. Understanding these protective factors helps identify safer conditions.
                </>
              )}
            </p>
          </div>

          <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
            <p className="text-xs text-slate-600 leading-relaxed flex items-start gap-2">
              <Info size={14} className="mt-0.5 flex-shrink-0" />
              <span>
                <span className="font-semibold">About SHAP:</span> SHAP (SHapley Additive exPlanations) 
                values represent each feature's contribution to the prediction. Positive values increase 
                severity, negative values decrease it. The magnitude indicates the strength of the effect. 
                {viewMode === 'global' ? ' Global importance shows average impact across all predictions.' : 
                ' Prediction-specific values show impact for the current input.'}
              </span>
            </p>
          </div>
        </div>
      </Card>

      {/* How to Use Guide */}
      <Card title="How to Use SHAP Analysis">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-slate-900 mb-3">Global Importance</h4>
            <ul className="space-y-2 text-sm text-slate-600">
              <li className="flex items-start gap-2">
                <span className="text-primary-600 mt-1">•</span>
                <span>Shows which features matter most across all predictions</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary-600 mt-1">•</span>
                <span>Helps understand overall model behavior</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary-600 mt-1">•</span>
                <span>Useful for feature selection and model improvement</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary-600 mt-1">•</span>
                <span>Based on average absolute SHAP values</span>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-semibold text-slate-900 mb-3">Prediction-Specific</h4>
            <ul className="space-y-2 text-sm text-slate-600">
              <li className="flex items-start gap-2">
                <span className="text-primary-600 mt-1">•</span>
                <span>Shows why a specific prediction was made</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary-600 mt-1">•</span>
                <span>Explains individual decisions in detail</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary-600 mt-1">•</span>
                <span>Red bars push severity up, blue bars push it down</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary-600 mt-1">•</span>
                <span>Uses current prediction parameters</span>
              </li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  )
}
