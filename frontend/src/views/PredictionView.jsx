import { useState } from 'react'
import { Target, AlertCircle, Users, Zap, Crown } from 'lucide-react'
import Card from '../components/ui/Card'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import axios from 'axios'

const LIGHTING_OPTIONS = {
  1: 'Daylight',
  2: 'Dusk/Dawn',
  3: 'Night with street lights',
  4: 'Night without street lights',
  5: 'Not specified'
}

const LOCATION_OPTIONS = {
  1: 'Urban area',
  2: 'Rural area'
}

const INTERSECTION_OPTIONS = {
  1: 'None',
  2: 'X intersection',
  3: 'T intersection',
  4: 'Y intersection',
  5: '5+ branches',
  6: 'Roundabout',
  7: 'Square',
  8: 'Level crossing',
  9: 'Other'
}

const DAY_OPTIONS = {
  0: 'Monday',
  1: 'Tuesday',
  2: 'Wednesday',
  3: 'Thursday',
  4: 'Friday',
  5: 'Saturday',
  6: 'Sunday'
}

const AVAILABLE_MODELS = [
  {
    id: 'stacking',
    name: 'Stacking Ensemble',
    icon: Zap,
    speed: 'Fast',
    isPro: false,
    badge: 'Recommended',
    badgeColor: 'bg-green-100 text-green-800',
    bestFor: 'Highest accuracy and reliability'
  },
  {
    id: 'xgboost_v2',
    name: 'XGBoost V2',
    icon: Crown,
    speed: 'Very Fast',
    isPro: true,
    badge: 'Pro',
    badgeColor: 'bg-purple-100 text-purple-800',
    bestFor: 'Fast predictions with high accuracy'
  },
  {
    id: 'xgboost_v1',
    name: 'XGBoost V1',
    icon: Target,
    speed: 'Very Fast',
    isPro: false,
    badge: null,
    badgeColor: null,
    bestFor: 'Quick predictions'
  },
  {
    id: 'random_forest_v2',
    name: 'Random Forest V2',
    icon: Crown,
    speed: 'Fast',
    isPro: true,
    badge: 'Pro',
    badgeColor: 'bg-purple-100 text-purple-800',
    bestFor: 'Understanding feature importance'
  },
  {
    id: 'random_forest_v1',
    name: 'Random Forest V1',
    icon: Target,
    speed: 'Fast',
    isPro: false,
    badge: null,
    badgeColor: null,
    bestFor: 'Stable predictions'
  },
  {
    id: 'tabtransformer',
    name: 'TabTransformer',
    icon: Target,
    speed: 'Medium',
    isPro: false,
    badge: 'Deep Learning',
    badgeColor: 'bg-blue-100 text-blue-800',
    bestFor: 'Complex pattern recognition'
  }
]

export default function PredictionView() {
  const [formData, setFormData] = useState({
    lighting: 1,
    location: 1,
    intersection: 1,
    day_of_week: 0,
    hour: 12,
    num_users: 2
  })
  const [selectedModel, setSelectedModel] = useState('stacking')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    
    try {
      const response = await axios.post('http://localhost:8000/api/predict', {
        ...formData,
        model: selectedModel  // Send selected model to backend
      })
      setResult(response.data)
    } catch (error) {
      console.error('Prediction failed:', error)
      setError(error.response?.data?.detail || 'Prediction failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleChange = (name, value) => {
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  const currentModel = AVAILABLE_MODELS.find(m => m.id === selectedModel)

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Model Selection */}
      <Card title="Choose Your Model" subtitle="Select the prediction model">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {AVAILABLE_MODELS.map((model) => {
            const Icon = model.icon
            const isSelected = selectedModel === model.id
            
            return (
              <button
                key={model.id}
                onClick={() => !model.isPro && setSelectedModel(model.id)}
                disabled={model.isPro}
                className={`relative p-4 rounded-lg border-2 transition-all text-left ${
                  model.isPro
                    ? 'border-slate-200 bg-slate-50 opacity-60 cursor-not-allowed'
                    : isSelected
                    ? 'border-primary-600 bg-primary-50 cursor-pointer'
                    : 'border-slate-200 hover:border-slate-300 bg-white cursor-pointer'
                }`}
              >
                {/* Badge */}
                {model.badge && (
                  <div className="absolute top-3 right-3">
                    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${model.badgeColor}`}>
                      {model.isPro && <Crown size={12} className="mr-1" />}
                      {model.badge}
                    </span>
                  </div>
                )}

                {/* Icon */}
                <div className={`inline-flex p-2 rounded-lg mb-3 ${
                  isSelected ? 'bg-primary-100' : 'bg-slate-100'
                }`}>
                  <Icon size={24} className={isSelected ? 'text-primary-600' : 'text-slate-600'} />
                </div>

                {/* Title */}
                <h3 className={`font-semibold mb-3 ${
                  isSelected ? 'text-primary-900' : 'text-slate-900'
                }`}>
                  {model.name}
                </h3>

                {/* Best For - without label */}
                <p className="text-sm text-slate-600 mb-3">
                  {model.bestFor}
                </p>

                {/* Speed */}
                <div className="text-xs text-slate-500">
                  <span className="font-semibold">Speed:</span>
                  <span className="ml-1">{model.speed}</span>
                </div>

                {/* Selected Indicator */}
                {isSelected && (
                  <div className="absolute inset-0 rounded-lg ring-2 ring-primary-600 pointer-events-none" />
                )}
              </button>
            )
          })}
        </div>

        {/* Pro Version Notice */}
        <div className="mt-4 p-3 bg-purple-50 border border-purple-200 rounded-lg flex items-start gap-2">
          <Crown size={16} className="text-purple-600 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-purple-900">
            <span className="font-semibold">Pro Models:</span> V2 optimized models are available in the Pro version with enhanced accuracy and performance.
          </p>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <Card title="Accident Parameters" subtitle="Configure the scenario details">
          {/* Selected Model Indicator */}
          <div className="mb-4 p-3 bg-slate-50 rounded-lg border border-slate-200">
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-600">Using model:</span>
              <span className="font-semibold text-slate-900">{currentModel?.name}</span>
              {currentModel?.isPro && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                  <Crown size={10} className="mr-1" />
                  Pro
                </span>
              )}
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Lighting Conditions
              </label>
              <select
                value={formData.lighting}
                onChange={(e) => handleChange('lighting', parseInt(e.target.value))}
                className="input-field"
              >
                {Object.entries(LIGHTING_OPTIONS).map(([code, label]) => (
                  <option key={code} value={code}>{label}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Location Type
              </label>
              <select
                value={formData.location}
                onChange={(e) => handleChange('location', parseInt(e.target.value))}
                className="input-field"
              >
                {Object.entries(LOCATION_OPTIONS).map(([code, label]) => (
                  <option key={code} value={code}>{label}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Intersection Type
              </label>
              <select
                value={formData.intersection}
                onChange={(e) => handleChange('intersection', parseInt(e.target.value))}
                className="input-field"
              >
                {Object.entries(INTERSECTION_OPTIONS).map(([code, label]) => (
                  <option key={code} value={code}>{label}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Day of Week
              </label>
              <select
                value={formData.day_of_week}
                onChange={(e) => handleChange('day_of_week', parseInt(e.target.value))}
                className="input-field"
              >
                {Object.entries(DAY_OPTIONS).map(([code, label]) => (
                  <option key={code} value={code}>{label}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Hour of Day
              </label>
              <div className="space-y-2">
                <input
                  type="range"
                  min={0}
                  max={23}
                  value={formData.hour}
                  onChange={(e) => handleChange('hour', parseInt(e.target.value))}
                  className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
                />
                <div className="flex justify-between text-sm text-slate-500">
                  <span>0</span>
                  <span className="font-medium text-slate-900">{formData.hour}:00</span>
                  <span>23</span>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Number of People Involved
              </label>
              <div className="space-y-2">
                <input
                  type="range"
                  min={1}
                  max={10}
                  value={formData.num_users}
                  onChange={(e) => handleChange('num_users', parseInt(e.target.value))}
                  className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
                />
                <div className="flex justify-between text-sm text-slate-500">
                  <span>1</span>
                  <span className="font-medium text-slate-900 flex items-center gap-1">
                    <Users size={14} />
                    {formData.num_users}
                  </span>
                  <span>10</span>
                </div>
              </div>
            </div>

            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <LoadingSpinner size="sm" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Target size={18} />
                  <span>Predict Collision Type</span>
                </>
              )}
            </button>
          </form>
        </Card>

        {/* Results Card */}
        <Card title="Prediction Results" subtitle={selectedModel === 'stacking' ? 'Ensemble model output' : `${currentModel?.name} output`}>
          {!result ? (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <AlertCircle size={48} className="text-slate-300 mb-4" />
              <p className="text-slate-500">
                Configure parameters and click "Predict Collision Type" to see results
              </p>
            </div>
          ) : selectedModel === 'stacking' ? (
            // Full ensemble view
            <div className="space-y-6">
              {/* Collision Type Prediction */}
              <div className="p-6 rounded-xl border-2 border-primary-200 bg-primary-50">
                <div className="text-center">
                  <p className="text-sm font-medium uppercase tracking-wide text-primary-700 mb-2">
                    Predicted Collision Type
                  </p>
                  <p className="text-3xl font-bold text-primary-900 mb-2">
                    {result.final_prediction.collision?.class_name || result.final_prediction.class_name}
                  </p>
                  <p className="text-sm text-primary-700">
                    Class {result.final_prediction.collision?.class || result.final_prediction.class}
                  </p>
                </div>
              </div>

              {/* Severity Prediction (if available) */}
              {result.final_prediction.severity && (
                <div className="p-6 rounded-xl border-2 border-orange-200 bg-orange-50">
                  <div className="text-center">
                    <p className="text-sm font-medium uppercase tracking-wide text-orange-700 mb-2">
                      Predicted Severity
                    </p>
                    <p className="text-3xl font-bold text-orange-900 mb-2">
                      {result.final_prediction.severity.class_name}
                    </p>
                    <p className="text-sm text-orange-700">
                      Level {result.final_prediction.severity.class}
                    </p>
                  </div>
                </div>
              )}

              {/* Model Agreement */}
              <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 text-center">
                <p className="text-sm text-slate-600 mb-1">Model Agreement</p>
                <p className="text-2xl font-bold text-slate-900">
                  {(result.ensemble.model_agreement * 100).toFixed(1)}%
                </p>
              </div>

              {/* Individual Models */}
              <div>
                <p className="text-sm font-semibold text-slate-700 mb-3">Individual Models</p>
                <div className="space-y-2">
                  {Object.entries(result.individual_models).map(([model, data]) => (
                    <div key={model} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                      <span className="text-sm font-medium text-slate-700">{model}</span>
                      <span className="text-sm text-slate-900">{data.prediction_name}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Reliability Indicator */}
              <div className={`p-4 rounded-lg border ${
                result.ensemble.model_agreement > 0.8 
                  ? 'bg-green-50 border-green-200' 
                  : result.ensemble.model_agreement > 0.6
                  ? 'bg-blue-50 border-blue-200'
                  : 'bg-yellow-50 border-yellow-200'
              }`}>
                <p className={`text-sm ${
                  result.ensemble.model_agreement > 0.8 
                    ? 'text-green-900' 
                    : result.ensemble.model_agreement > 0.6
                    ? 'text-blue-900'
                    : 'text-yellow-900'
                }`}>
                  <span className="font-semibold">
                    {result.ensemble.model_agreement > 0.8 
                      ? '✓ High Agreement' 
                      : result.ensemble.model_agreement > 0.6
                      ? 'ℹ Moderate Agreement'
                      : '⚠ Low Agreement'}
                  </span>
                  {' - '}
                  Variance: {result.ensemble.variance.toFixed(4)}
                </p>
              </div>
            </div>
          ) : (
            // Simplified single model view
            <div className="space-y-6">
              {/* Collision Type Prediction */}
              <div className="p-8 rounded-xl border-2 border-primary-200 bg-gradient-to-br from-primary-50 to-blue-50">
                <div className="text-center">
                  <p className="text-sm font-medium uppercase tracking-wide text-primary-700 mb-3">
                    Predicted Collision Type
                  </p>
                  <p className="text-5xl font-bold text-primary-900 mb-3">
                    {result.final_prediction.collision?.class_name || result.final_prediction.class_name}
                  </p>
                  <p className="text-lg text-primary-700">
                    Class {result.final_prediction.collision?.class || result.final_prediction.class}
                  </p>
                </div>
              </div>

              {/* Severity Prediction (if available) */}
              {result.final_prediction.severity && (
                <div className="p-6 rounded-xl border-2 border-orange-200 bg-gradient-to-br from-orange-50 to-red-50">
                  <div className="text-center">
                    <p className="text-sm font-medium uppercase tracking-wide text-orange-700 mb-2">
                      Predicted Severity
                    </p>
                    <p className="text-4xl font-bold text-orange-900 mb-2">
                      {result.final_prediction.severity.class_name}
                    </p>
                    <p className="text-sm text-orange-700">
                      Level {result.final_prediction.severity.level || (result.final_prediction.severity.class + 1)}
                    </p>
                  </div>
                </div>
              )}

              {/* Model Info */}
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-900">
                  <span className="font-semibold">Model:</span> {currentModel?.name}
                  <br />
                  <span className="font-semibold">Optimized for:</span> {currentModel?.bestFor}
                </p>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
