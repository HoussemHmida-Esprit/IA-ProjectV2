"""
Configuration Management - Single Source of Truth
Follows 12-Factor App principles
"""
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
import os


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    
    # Categorical features with valid ranges
    CATEGORICAL_FEATURES: Dict[str, tuple] = field(default_factory=lambda: {
        'lum': (1, 5),  # Lighting: 1=Day, 2=Twilight, 3=Night(lit), 4=Night(unlit), 5=Unknown
        'atm': (1, 9),  # Weather: 1-9 different conditions
        'agg': (1, 2),  # Location: 1=Rural, 2=Urban
        'int': (1, 9),  # Intersection: 1-9 types
    })
    
    # Numerical features with valid ranges
    NUMERICAL_FEATURES: Dict[str, tuple] = field(default_factory=lambda: {
        'hour': (0, 23),
        'day_of_week': (0, 6),
        'month': (1, 12),
        'num_users': (1, 100),  # Reasonable max
    })
    
    # Target variables with valid ranges
    TARGET_RANGES: Dict[str, tuple] = field(default_factory=lambda: {
        'col': (1, 7),  # Collision type: 1-7
        'max_severity': (1, 4),  # Severity: 1-4
    })
    
    # Feature selection
    ALL_FEATURES: List[str] = field(default_factory=lambda: [
        'lum', 'atm', 'agg', 'int', 'hour', 'day_of_week', 'month', 'num_users'
    ])


@dataclass
class ModelConfig:
    """Model hyperparameters and settings"""
    
    # Random Forest
    RF_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    })
    
    # XGBoost
    XGB_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    })
    
    # Training
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5


@dataclass
class PathConfig:
    """File paths and directories"""
    
    # Base paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # Data files
    RAW_DATA: Path = DATA_DIR / "cleaned_accidents.csv"
    MODEL_READY: Path = DATA_DIR / "model_ready.csv"
    
    # Model files
    RF_MODEL: Path = MODELS_DIR / "rf_production.pkl"
    XGB_MODEL: Path = MODELS_DIR / "xgb_production.pkl"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.LOGS_DIR.mkdir(exist_ok=True, parents=True)
        self.MODELS_DIR.mkdir(exist_ok=True, parents=True)


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    
    # Performance thresholds
    MIN_ACCURACY: float = 0.70  # Alert if accuracy drops below 70%
    MAX_PREDICTION_TIME_MS: float = 100.0  # Alert if prediction takes > 100ms
    
    # Drift detection
    DRIFT_THRESHOLD: float = 0.1  # Alert if feature distribution shifts > 10%
    DRIFT_CHECK_INTERVAL_DAYS: int = 7
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class ValidationConfig:
    """Data validation rules"""
    
    # Missing value thresholds
    MAX_MISSING_RATE: float = 0.1  # Reject if > 10% missing
    
    # Outlier detection
    OUTLIER_STD_THRESHOLD: float = 3.0  # Flag values > 3 std devs
    
    # Business rules
    BUSINESS_RULES: Dict = field(default_factory=lambda: {
        'fatal_low_speed': {
            'condition': lambda row: row['max_severity'] == 4 and row.get('speed', 50) < 20,
            'message': 'Fatal accident at low speed - verify data'
        },
        'no_collision_with_injury': {
            'condition': lambda row: row['col'] == 7 and row.get('num_users', 1) > 1,
            'message': 'No collision but multiple users - verify data'
        }
    })


# Global configuration instances
FEATURES = FeatureConfig()
MODELS = ModelConfig()
PATHS = PathConfig()
MONITORING = MonitoringConfig()
VALIDATION = ValidationConfig()
