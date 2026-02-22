# Accident Severity Predictor - Frontend

A professional, minimalist React application for ML-powered accident severity prediction with explainable AI.

## Tech Stack

- **React 18** with Vite
- **Tailwind CSS** for styling
- **React Router** for navigation
- **Recharts** for data visualization
- **Lucide React** for icons
- **Axios** for API calls

## Getting Started

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   └── AppLayout.jsx       # Main layout with sidebar
│   │   └── ui/
│   │       ├── Card.jsx            # Reusable card component
│   │       ├── LoadingSpinner.jsx  # Loading indicator
│   │       └── StatusIndicator.jsx # System status badges
│   ├── views/
│   │   ├── OverviewDashboard.jsx   # Main dashboard
│   │   ├── PredictionView.jsx      # Live prediction interface
│   │   └── XaiDashboard.jsx        # SHAP explainability
│   ├── App.jsx                     # Root component
│   ├── main.jsx                    # Entry point
│   └── index.css                   # Global styles
├── index.html
├── package.json
├── tailwind.config.js
└── vite.config.js
```

## Features

### 1. Overview Dashboard
- Real-time system metrics
- Model performance trends
- Quick action buttons
- System status indicators

### 2. Live Prediction Engine
- Interactive form with dropdowns and sliders
- Real-time severity prediction
- Confidence scores
- Clean results visualization

### 3. Explainable AI (XAI)
- SHAP force plot visualization
- Feature importance ranking
- Plain-language explanations
- Color-coded impact indicators

## API Integration

The app is configured to proxy API requests to `http://localhost:8000`. Update `vite.config.js` if your backend runs on a different port.

### Expected API Endpoints

```javascript
// Prediction
POST /api/predict
Body: {
  lighting: string,
  weather: string,
  intersection: string,
  hour: number,
  speed_limit: number,
  num_vehicles: number
}

// SHAP Values
GET /api/shap
Response: [
  { feature: string, value: number }
]
```

## Mock Data

The application includes mock data for immediate testing. To connect to your FastAPI backend:

1. Uncomment the axios calls in `PredictionView.jsx`
2. Remove the mock response simulation
3. Ensure your backend is running on port 8000

## Design System

- **Colors**: Deep slate (#1a202e) and electric blue (#2563eb)
- **Typography**: Inter font family
- **Spacing**: Generous whitespace with 8px base unit
- **Components**: Shadcn UI-inspired patterns
- **Icons**: Lucide React icon set

## Customization

### Changing Colors

Edit `tailwind.config.js`:

```javascript
colors: {
  primary: {
    600: '#your-color',
    700: '#your-darker-color',
  }
}
```

### Adding New Routes

1. Create a new view in `src/views/`
2. Add route to `src/App.jsx`
3. Add navigation item to `src/components/layout/AppLayout.jsx`
