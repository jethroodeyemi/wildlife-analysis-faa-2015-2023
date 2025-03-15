"""
Machine learning models for predicting wildlife strike risks
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
from typing import Tuple, Dict, List, Any

class WildlifeStrikeRiskPredictor:
    def __init__(self):
        self.damage_classifier = None
        self.cost_predictor = None
        self.feature_preprocessor = None
        
    def prepare_features(self, df: pd.DataFrame) -> None:
        """
        Prepare feature preprocessing pipeline
        """
        numeric_features = [
            'HEIGHT', 'SPEED', 'DISTANCE', 'AC_MASS', 'NUM_ENGS'
        ]
        
        categorical_features = [
            'OPERATOR', 'AC_CLASS', 'TYPE_ENG', 'TIME_OF_DAY', 'SPECIES'
        ]
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.feature_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
    def train_damage_classifier(self, 
                              X: pd.DataFrame, 
                              y: pd.Series,
                              **kwargs) -> Dict[str, float]:
        """
        Train a classifier to predict damage occurrence
        """
        # Create pipeline
        self.damage_classifier = Pipeline([
            ('preprocessor', self.feature_preprocessor),
            ('classifier', RandomForestClassifier(**kwargs))
        ])
        
        # Train model
        self.damage_classifier.fit(X, y)
        
        # Evaluate model
        cv_scores = cross_val_score(
            self.damage_classifier, X, y, cv=5, scoring='f1'
        )
        
        return {
            'mean_f1': cv_scores.mean(),
            'std_f1': cv_scores.std()
        }
    
    def train_cost_predictor(self,
                           X: pd.DataFrame,
                           y: pd.Series,
                           **kwargs) -> Dict[str, float]:
        """
        Train a regressor to predict strike costs
        """
        # Create pipeline
        self.cost_predictor = Pipeline([
            ('preprocessor', self.feature_preprocessor),
            ('regressor', GradientBoostingRegressor(**kwargs))
        ])
        
        # Train model
        self.cost_predictor.fit(X, y)
        
        # Evaluate model
        cv_scores = cross_val_score(
            self.cost_predictor, X, y, cv=5, scoring='r2'
        )
        
        return {
            'mean_r2': cv_scores.mean(),
            'std_r2': cv_scores.std()
        }
    
    def predict_risk_factors(self,
                           X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict both damage probability and expected cost
        """
        damage_prob = self.damage_classifier.predict_proba(X)[:, 1]
        expected_cost = self.cost_predictor.predict(X)
        
        return {
            'damage_probability': damage_prob,
            'expected_cost': expected_cost,
            'risk_score': damage_prob * np.log1p(expected_cost)
        }
    
    def get_feature_importance(self) -> Dict[str, pd.Series]:
        """
        Get feature importance for both models
        """
        # Get feature names after preprocessing
        feature_names = (
            self.feature_preprocessor.get_feature_names_out()
        )
        
        # Get importance scores
        damage_importance = pd.Series(
            self.damage_classifier.named_steps['classifier'].feature_importances_,
            index=feature_names
        )
        
        cost_importance = pd.Series(
            self.cost_predictor.named_steps['regressor'].feature_importances_,
            index=feature_names
        )
        
        return {
            'damage_model': damage_importance.sort_values(ascending=False),
            'cost_model': cost_importance.sort_values(ascending=False)
        }
    
    def save_models(self, path: str) -> None:
        """
        Save trained models to disk
        """
        joblib.dump(self.damage_classifier, f"{path}/damage_classifier.joblib")
        joblib.dump(self.cost_predictor, f"{path}/cost_predictor.joblib")
    
    def load_models(self, path: str) -> None:
        """
        Load trained models from disk
        """
        self.damage_classifier = joblib.load(f"{path}/damage_classifier.joblib")
        self.cost_predictor = joblib.load(f"{path}/cost_predictor.joblib")

def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for model training
    """
    # Select features
    features = [
        'HEIGHT', 'SPEED', 'DISTANCE', 'AC_MASS', 'NUM_ENGS',
        'AC_CLASS', 'TYPE_ENG', 'TIME_OF_DAY', 'SPECIES',
        'OPERATOR'
    ]
    
    X = df[features]
    df['TOTAL_COST'] = df[['COST_REPAIRS_INFL_ADJ', 'COST_OTHER_INFL_ADJ']].sum(axis=1)
    y_damage = df['INDICATED_DAMAGE']
    y_cost = df['TOTAL_COST']
    
    return X, y_damage, y_cost