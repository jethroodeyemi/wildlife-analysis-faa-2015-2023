"""
Data validation utilities for wildlife strikes analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

def check_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.1
) -> Dict[str, float]:
    """
    Check for missing values in each column
    Returns columns with missing value rates above threshold
    """
    missing_rates = df.isnull().mean()
    problematic_columns = missing_rates[missing_rates > threshold]
    
    return {
        col: rate 
        for col, rate in problematic_columns.items()
    }

def validate_date_ranges(
    df: pd.DataFrame,
    date_column: str = 'INCIDENT_DATE',
    min_date: Optional[str] = None,
    max_date: Optional[str] = None
) -> Dict[str, List[pd.Timestamp]]:
    """
    Validate date ranges in the dataset
    """
    dates = pd.to_datetime(df[date_column])
    invalid_dates = []
    
    # Check for dates in the future
    future_dates = dates[dates > datetime.now()]
    
    # Check for dates before minimum allowed date
    if min_date:
        min_allowed = pd.to_datetime(min_date)
        early_dates = dates[dates < min_allowed]
    else:
        early_dates = pd.Series([], dtype='datetime64[ns]')
    
    # Check for dates after maximum allowed date
    if max_date:
        max_allowed = pd.to_datetime(max_date)
        late_dates = dates[dates > max_allowed]
    else:
        late_dates = pd.Series([], dtype='datetime64[ns]')
    
    return {
        'future_dates': future_dates.tolist(),
        'early_dates': early_dates.tolist(),
        'late_dates': late_dates.tolist()
    }

def validate_numeric_ranges(
    df: pd.DataFrame,
    column_ranges: Dict[str, Tuple[float, float]]
) -> Dict[str, pd.Series]:
    """
    Validate numeric values are within expected ranges
    """
    out_of_range = {}
    
    for column, (min_val, max_val) in column_ranges.items():
        if column in df.columns:
            invalid_values = df[
                (df[column] < min_val) | 
                (df[column] > max_val)
            ][column]
            
            if not invalid_values.empty:
                out_of_range[column] = invalid_values
    
    return out_of_range

def check_data_consistency(
    df: pd.DataFrame
) -> Dict[str, List[int]]:
    """
    Check for logical consistency in data
    """
    inconsistencies = {}
    
    # Check cost consistency
    if all(col in df.columns for col in ['TOTAL_COST', 'COST_REPAIRS', 'COST_OTHER']):
        cost_mismatch = df[
            abs(
                df['TOTAL_COST'] - 
                (df['COST_REPAIRS'] + df['COST_OTHER'])
            ) > 1  # Allow for small rounding differences
        ].index.tolist()
        
        if cost_mismatch:
            inconsistencies['cost_mismatch'] = cost_mismatch
    
    # Check damage score consistency
    if 'DAMAGE' in df.columns and 'DAMAGE_SCORE' in df.columns:
        damage_mismatch = df[
            (df['DAMAGE'] == 'N') & 
            (df['DAMAGE_SCORE'] > 0)
        ].index.tolist()
        
        if damage_mismatch:
            inconsistencies['damage_mismatch'] = damage_mismatch
    
    # Check injury count consistency
    if all(col in df.columns for col in ['NR_INJURIES', 'NR_FATALITIES']):
        injury_mismatch = df[
            df['NR_FATALITIES'] > df['NR_INJURIES']
        ].index.tolist()
        
        if injury_mismatch:
            inconsistencies['injury_mismatch'] = injury_mismatch
    
    return inconsistencies

def validate_categorical_values(
    df: pd.DataFrame,
    valid_categories: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Check for invalid categorical values
    """
    invalid_values = {}
    
    for column, valid_values in valid_categories.items():
        if column in df.columns:
            invalid = df[
                ~df[column].isin(valid_values) & 
                df[column].notna()
            ][column].unique().tolist()
            
            if invalid:
                invalid_values[column] = invalid
    
    return invalid_values

def check_duplicate_records(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None
) -> pd.Index:
    """
    Check for duplicate records
    """
    if subset is None:
        duplicates = df[df.duplicated(keep='first')].index
    else:
        duplicates = df[df.duplicated(subset=subset, keep='first')].index
    
    return duplicates

def validate_relational_integrity(
    df: pd.DataFrame,
    reference_data: Dict[str, pd.DataFrame],
    foreign_keys: Dict[str, Tuple[str, str]]
) -> Dict[str, List]:
    """
    Validate foreign key relationships
    """
    integrity_violations = {}
    
    for column, (ref_table, ref_column) in foreign_keys.items():
        if column in df.columns and ref_table in reference_data:
            valid_values = set(reference_data[ref_table][ref_column])
            invalid_refs = df[
                ~df[column].isin(valid_values) & 
                df[column].notna()
            ][column].unique().tolist()
            
            if invalid_refs:
                integrity_violations[column] = invalid_refs
    
    return integrity_violations

def generate_validation_report(
    df: pd.DataFrame,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    column_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    valid_categories: Optional[Dict[str, List[str]]] = None,
    reference_data: Optional[Dict[str, pd.DataFrame]] = None,
    foreign_keys: Optional[Dict[str, Tuple[str, str]]] = None
) -> Dict:
    """
    Generate comprehensive data validation report
    """
    report = {
        'summary': {
            'total_records': len(df),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Check missing values
    missing = check_missing_values(df)
    if missing:
        report['missing_values'] = missing
    
    # Validate dates
    date_issues = validate_date_ranges(df, min_date=min_date, max_date=max_date)
    if any(date_issues.values()):
        report['date_issues'] = date_issues
    
    # Validate numeric ranges
    if column_ranges:
        range_issues = validate_numeric_ranges(df, column_ranges)
        if range_issues:
            report['range_violations'] = range_issues
    
    # Check consistency
    consistency_issues = check_data_consistency(df)
    if consistency_issues:
        report['consistency_issues'] = consistency_issues
    
    # Validate categories
    if valid_categories:
        category_issues = validate_categorical_values(df, valid_categories)
        if category_issues:
            report['invalid_categories'] = category_issues
    
    # Check duplicates
    duplicates = check_duplicate_records(df)
    if not duplicates.empty:
        report['duplicate_records'] = duplicates.tolist()
    
    # Validate relationships
    if reference_data and foreign_keys:
        integrity_issues = validate_relational_integrity(
            df, reference_data, foreign_keys
        )
        if integrity_issues:
            report['integrity_violations'] = integrity_issues
    
    return report

def validate_time_series_continuity(
    df: pd.DataFrame,
    date_column: str = 'INCIDENT_DATE',
    freq: str = 'D'
) -> Dict[str, List[pd.Timestamp]]:
    """
    Check for gaps in time series data
    """
    dates = pd.to_datetime(df[date_column])
    date_range = pd.date_range(
        start=dates.min(),
        end=dates.max(),
        freq=freq
    )
    
    missing_dates = date_range[~date_range.isin(dates)].tolist()
    
    return {
        'missing_dates': missing_dates,
        'gap_starts': [
            date for i, date in enumerate(missing_dates[:-1])
            if (missing_dates[i+1] - date).days > 1
        ]
    }

def validate_geographic_coordinates(
    df: pd.DataFrame,
    lat_column: str = 'AIRPORT_LATITUDE',
    lon_column: str = 'AIRPORT_LONGITUDE'
) -> Dict[str, List[int]]:
    """
    Validate geographic coordinates
    """
    invalid_coordinates = {
        'invalid_latitude': df[
            (df[lat_column] < -90) | 
            (df[lat_column] > 90)
        ].index.tolist(),
        
        'invalid_longitude': df[
            (df[lon_column] < -180) | 
            (df[lon_column] > 180)
        ].index.tolist()
    }
    
    return invalid_coordinates