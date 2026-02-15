"""
Data Validation Layer - Catch Bad Data Before It Hits Models
Uses Pydantic for schema validation
"""
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, validator, ValidationError
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from ..config import FEATURES, VALIDATION

logger = logging.getLogger(__name__)


class AccidentRecord(BaseModel):
    """
    Schema for a single accident record
    Enforces type safety and business rules
    """
    
    # Categorical features with validation
    lum: int = Field(..., ge=1, le=5, description="Lighting conditions")
    atm: Optional[int] = Field(None, ge=1, le=9, description="Weather conditions")
    agg: int = Field(..., ge=1, le=2, description="Urban/Rural")
    int_type: int = Field(..., ge=1, le=9, alias='int', description="Intersection type")
    
    # Numerical features with validation
    hour: int = Field(..., ge=0, le=23, description="Hour of day")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week")
    month: int = Field(..., ge=1, le=12, description="Month")
    num_users: int = Field(..., ge=1, le=100, description="Number of people involved")
    
    # Optional target (for training data)
    col: Optional[int] = Field(None, ge=1, le=7, description="Collision type")
    max_severity: Optional[int] = Field(None, ge=1, le=4, description="Max severity")
    
    class Config:
        # Allow field aliases (int -> int_type)
        allow_population_by_field_name = True
    
    @validator('hour')
    def validate_hour(cls, v):
        """Ensure hour is valid"""
        if not 0 <= v <= 23:
            raise ValueError(f"Hour must be 0-23, got {v}")
        return v
    
    @validator('num_users')
    def validate_num_users(cls, v):
        """Ensure reasonable number of users"""
        if v < 1:
            raise ValueError("At least 1 user required")
        if v > 100:
            logger.warning(f"Unusually high number of users: {v}")
        return v
    
    @validator('col', 'max_severity')
    def validate_targets(cls, v, field):
        """Validate target variables if present"""
        if v is not None:
            if field.name == 'col' and not 1 <= v <= 7:
                raise ValueError(f"Collision type must be 1-7, got {v}")
            if field.name == 'max_severity' and not 1 <= v <= 4:
                raise ValueError(f"Severity must be 1-4, got {v}")
        return v


class DataValidator:
    """
    Comprehensive data validation pipeline
    Catches issues before they reach the model
    """
    
    def __init__(self):
        self.validation_errors: List[Dict] = []
        self.warnings: List[Dict] = []
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        strict: bool = True
    ) -> Tuple[bool, pd.DataFrame, List[Dict]]:
        """
        Validate entire dataframe
        
        Args:
            df: Input dataframe
            strict: If True, raise on any error. If False, log and continue
            
        Returns:
            (is_valid, cleaned_df, errors)
        """
        logger.info(f"Validating dataframe with {len(df)} records")
        
        self.validation_errors = []
        self.warnings = []
        
        # 1. Schema validation
        valid_df, schema_errors = self._validate_schema(df)
        
        # 2. Missing value check
        missing_errors = self._check_missing_values(valid_df)
        
        # 3. Range validation
        range_errors = self._validate_ranges(valid_df)
        
        # 4. Business rules
        business_errors = self._check_business_rules(valid_df)
        
        # 5. Statistical outliers
        outlier_warnings = self._detect_outliers(valid_df)
        
        # Combine all errors
        all_errors = (
            schema_errors + 
            missing_errors + 
            range_errors + 
            business_errors
        )
        
        self.validation_errors = all_errors
        self.warnings = outlier_warnings
        
        # Log summary
        if all_errors:
            logger.error(f"Validation failed with {len(all_errors)} errors")
            for error in all_errors[:5]:  # Show first 5
                logger.error(f"  - {error}")
        
        if outlier_warnings:
            logger.warning(f"Found {len(outlier_warnings)} outliers")
        
        is_valid = len(all_errors) == 0
        
        if not is_valid and strict:
            raise ValidationError(f"Data validation failed: {all_errors}")
        
        return is_valid, valid_df, all_errors
    
    def _validate_schema(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Validate each record against Pydantic schema"""
        errors = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            try:
                # Validate record
                AccidentRecord(**row.to_dict())
                valid_indices.append(idx)
            except ValidationError as e:
                errors.append({
                    'row': idx,
                    'type': 'schema_error',
                    'details': str(e)
                })
        
        # Return only valid rows
        valid_df = df.loc[valid_indices]
        
        if errors:
            logger.warning(f"Schema validation: {len(errors)} invalid records removed")
        
        return valid_df, errors
    
    def _check_missing_values(self, df: pd.DataFrame) -> List[Dict]:
        """Check for excessive missing values"""
        errors = []
        
        for col in FEATURES.ALL_FEATURES:
            if col not in df.columns:
                errors.append({
                    'type': 'missing_column',
                    'column': col,
                    'message': f"Required column '{col}' not found"
                })
                continue
            
            missing_rate = df[col].isna().mean()
            if missing_rate > VALIDATION.MAX_MISSING_RATE:
                errors.append({
                    'type': 'excessive_missing',
                    'column': col,
                    'missing_rate': missing_rate,
                    'threshold': VALIDATION.MAX_MISSING_RATE,
                    'message': f"Column '{col}' has {missing_rate:.1%} missing (threshold: {VALIDATION.MAX_MISSING_RATE:.1%})"
                })
        
        return errors
    
    def _validate_ranges(self, df: pd.DataFrame) -> List[Dict]:
        """Validate feature values are within expected ranges"""
        errors = []
        
        # Check categorical features
        for feat, (min_val, max_val) in FEATURES.CATEGORICAL_FEATURES.items():
            if feat not in df.columns:
                continue
            
            invalid_mask = (df[feat] < min_val) | (df[feat] > max_val)
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                errors.append({
                    'type': 'range_violation',
                    'column': feat,
                    'expected_range': (min_val, max_val),
                    'invalid_count': invalid_count,
                    'invalid_values': df.loc[invalid_mask, feat].unique().tolist()[:10]
                })
        
        # Check numerical features
        for feat, (min_val, max_val) in FEATURES.NUMERICAL_FEATURES.items():
            if feat not in df.columns:
                continue
            
            invalid_mask = (df[feat] < min_val) | (df[feat] > max_val)
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                errors.append({
                    'type': 'range_violation',
                    'column': feat,
                    'expected_range': (min_val, max_val),
                    'invalid_count': invalid_count
                })
        
        return errors
    
    def _check_business_rules(self, df: pd.DataFrame) -> List[Dict]:
        """Apply business logic validation"""
        errors = []
        
        for rule_name, rule in VALIDATION.BUSINESS_RULES.items():
            try:
                violations = df.apply(rule['condition'], axis=1)
                violation_count = violations.sum()
                
                if violation_count > 0:
                    self.warnings.append({
                        'type': 'business_rule_violation',
                        'rule': rule_name,
                        'message': rule['message'],
                        'count': violation_count,
                        'severity': 'warning'
                    })
            except Exception as e:
                logger.error(f"Error checking business rule '{rule_name}': {e}")
        
        return errors
    
    def _detect_outliers(self, df: pd.DataFrame) -> List[Dict]:
        """Detect statistical outliers"""
        warnings = []
        
        for feat in FEATURES.NUMERICAL_FEATURES.keys():
            if feat not in df.columns:
                continue
            
            values = df[feat].dropna()
            if len(values) == 0:
                continue
            
            mean = values.mean()
            std = values.std()
            
            if std == 0:
                continue
            
            # Z-score method
            z_scores = np.abs((values - mean) / std)
            outliers = z_scores > VALIDATION.OUTLIER_STD_THRESHOLD
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                warnings.append({
                    'type': 'statistical_outlier',
                    'column': feat,
                    'count': outlier_count,
                    'percentage': outlier_count / len(values),
                    'threshold': VALIDATION.OUTLIER_STD_THRESHOLD
                })
        
        return warnings
    
    def get_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_errors': len(self.validation_errors),
            'total_warnings': len(self.warnings),
            'errors': self.validation_errors,
            'warnings': self.warnings,
            'status': 'PASS' if len(self.validation_errors) == 0 else 'FAIL'
        }
