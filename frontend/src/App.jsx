import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import AppLayout from './components/layout/AppLayout'
import OverviewDashboard from './views/OverviewDashboard'
import PredictionView from './views/PredictionView'
import XaiDashboard from './views/XaiDashboard'
import ForecastingView from './views/ForecastingView'

function App() {
  return (
    <Router>
      <AppLayout>
        <Routes>
          <Route path="/" element={<OverviewDashboard />} />
          <Route path="/predict" element={<PredictionView />} />
          <Route path="/xai" element={<XaiDashboard />} />
          <Route path="/forecast" element={<ForecastingView />} />
        </Routes>
      </AppLayout>
    </Router>
  )
}

export default App
