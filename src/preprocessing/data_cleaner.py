"""
Data cleaning and preprocessing utilities for FAA Wildlife Strikes Analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict

def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize date-related columns.
    """
    df = df.copy()
    # Convert date columns
    df['INCIDENT_DATE'] = pd.to_datetime(df['INCIDENT_DATE'])
    df['LUPDATE'] = pd.to_datetime(df['LUPDATE'])
    
    # Extract time components
    df['YEAR'] = df['INCIDENT_DATE'].dt.year
    df['MONTH'] = df['INCIDENT_DATE'].dt.month
    df['DAY'] = df['INCIDENT_DATE'].dt.day
    df['WEEKDAY'] = df['INCIDENT_DATE'].dt.day_name()
    df['SEASON'] = df['MONTH'].map(lambda x: 
        'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else 'Fall')
    
    return df

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean numeric columns, handling missing values and converting data types.
    """
    df = df.copy()
    
    # Cost-related columns
    cost_columns = ['COST_REPAIRS', 'COST_OTHER', 'COST_REPAIRS_INFL_ADJ', 'COST_OTHER_INFL_ADJ']
    for col in cost_columns:
        df[col] = pd.to_numeric(df[col].replace('', np.nan), errors='coerce')
    
    # Flight metrics
    df['HEIGHT'] = pd.to_numeric(df['HEIGHT'], errors='coerce')
    df['SPEED'] = pd.to_numeric(df['SPEED'], errors='coerce')
    df['DISTANCE'] = pd.to_numeric(df['DISTANCE'], errors='coerce')
    
    # Aircraft metrics
    df['NUM_ENGS'] = pd.to_numeric(df['NUM_ENGS'], errors='coerce')
    df['AC_MASS'] = pd.to_numeric(df['AC_MASS'], errors='coerce')
    
    return df

def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize categorical columns.
    """
    df = df.copy()
    
    # Standardize damage codes
    df['DAMAGE_LEVEL'] = df['DAMAGE_LEVEL'].fillna('').str.strip().str.upper()
    
    # Clean operator types
    df['OPERATOR'] = df['OPERATOR'].fillna('UNKNOWN')
    df['OPERATOR_CATEGORY'] = df['OPERATOR'].map(lambda x: 
        'COMMERCIAL' if x not in ['BUS', 'PVT', 'GOV', 'MIL'] else x)
    
    # Standardize wildlife information
    df['SPECIES'] = df['SPECIES'].fillna('UNKNOWN').str.strip().str.title()
    
    # Clean and categorize time of day
    df['TIME_OF_DAY'] = df['TIME_OF_DAY'].fillna('UNKNOWN').str.strip().str.upper()
    
    return df

def clean_location_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize location-related data.
    """
    df = df.copy()
    
    # Clean airport information
    df['AIRPORT'] = df['AIRPORT'].fillna('UNKNOWN').str.strip().str.upper()
    df['STATE'] = df['STATE'].fillna('UNKNOWN').str.strip().str.upper()
    
    # Handle coordinates
    df['AIRPORT_LATITUDE'] = pd.to_numeric(df['AIRPORT_LATITUDE'], errors='coerce')
    df['AIRPORT_LONGITUDE'] = pd.to_numeric(df['AIRPORT_LONGITUDE'], errors='coerce')
    
    # Clean location description
    df['LOCATION'] = df['LOCATION'].fillna('UNKNOWN').str.strip()
    
    return df

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for analysis.
    """
    df = df.copy()
    
    # Total cost calculation
    df['TOTAL_COST'] = df['COST_REPAIRS_INFL_ADJ'] + df['COST_OTHER_INFL_ADJ']
    
    # Severity score (based on damage and costs)
    damage_scores = {'': 0, 'N': 0, 'M': 1, 'M?': 1, 'S': 2, 'D': 3}
    df['DAMAGE_SCORE'] = df['DAMAGE_LEVEL'].map(damage_scores)
    
    # Calculate time between incident and report
    df['REPORT_DELAY_DAYS'] = (pd.to_datetime(df['LUPDATE']) - 
                              pd.to_datetime(df['INCIDENT_DATE'])).dt.days
    
    # Create impact indicators
    df['HAS_DAMAGE'] = df['DAMAGE_LEVEL'].isin(['M', 'M?', 'S', 'D'])
    df['HIGH_COST'] = df['TOTAL_COST'] > df['TOTAL_COST'].median()
    
    return df

def process_wildlife_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to process and clean the wildlife strikes dataset.
    """
    df = (df.pipe(clean_dates)
            .pipe(clean_numeric_columns)
            .pipe(clean_categorical_columns)
            .pipe(clean_location_data)
            .pipe(create_derived_features))
    
    return df