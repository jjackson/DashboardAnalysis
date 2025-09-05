"""
Data utilities for fake data analysis project.
Provides clean, standardized data loading and preprocessing functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime
import re

def get_most_recent_file(directory, prefix):
    """Get the most recent file with the given prefix based on timestamp."""
    pattern = f"{directory}/{prefix}-*.csv"
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No files found with prefix '{prefix}' in {directory}")
    
    # Extract timestamps and find most recent
    file_times = []
    for file in files:
        filename = Path(file).name
        timestamp_match = re.search(r'(\d{8}T\d{6})', filename)
        if timestamp_match:
            try:
                timestamp_str = timestamp_match.group(1)
                timestamp = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
                file_times.append((file, timestamp))
            except ValueError:
                continue
    
    if not file_times:
        return files[0]  # Fallback to first file
    
    return max(file_times, key=lambda x: x[1])[0]

def load_raw_data(file_path, sample_size=None):
    """
    Load raw CSV data with basic type inference.
    
    Args:
        file_path: Path to CSV file
        sample_size: Number of rows to load (None for all)
    
    Returns:
        pandas.DataFrame: Raw data
    """
    kwargs = {'nrows': sample_size} if sample_size else {}
    df = pd.read_csv(file_path, **kwargs)
    
    # Basic datetime conversion with flexible parsing
    if 'visit_date' in df.columns:
        df['visit_date'] = pd.to_datetime(df['visit_date'], format='mixed', errors='coerce')
    
    if 'visit_start_time' in df.columns:
        df['visit_start_time'] = pd.to_datetime(df['visit_start_time'], format='mixed', errors='coerce')
    
    if 'visit_end_time' in df.columns:
        df['visit_end_time'] = pd.to_datetime(df['visit_end_time'], format='mixed', errors='coerce')
    
    return df

def standardize_columns(df, data_type):
    """
    Standardize column names and add derived fields.
    
    Args:
        df: Raw dataframe
        data_type: 'real' or 'fake' to handle schema differences
    
    Returns:
        pandas.DataFrame: Standardized dataframe
    """
    df = df.copy()
    
    # Add data source identifier
    df['data_source'] = data_type
    
    # Standardize numeric columns
    numeric_cols = ['childs_age_in_month', 'soliciter_muac_cm', 'no_of_children']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add derived temporal features
    if 'visit_date' in df.columns:
        df['visit_hour'] = df['visit_date'].dt.hour
        df['visit_day_of_week'] = df['visit_date'].dt.dayofweek
        df['visit_month'] = df['visit_date'].dt.month
    
    # Calculate visit duration for fake data
    if data_type == 'fake' and 'visit_start_time' in df.columns and 'visit_end_time' in df.columns:
        df['visit_duration_minutes'] = (
            df['visit_end_time'] - df['visit_start_time']
        ).dt.total_seconds() / 60
    
    # Standardize categorical values
    categorical_mappings = {
        'childs_gender': {'male_child': 'male', 'female_child': 'female'},
        'muac_colour': {'Green': 'green', 'Yellow': 'yellow', 'Red': 'red'},
        'diagnosed_with_mal_past_3_months': {'yes': True, 'no': False},
        'under_treatment_for_mal': {'yes': True, 'no': False},
        'received_any_vaccine': {'yes': True, 'no': False},
        'muac_consent': {'yes': True, 'no': False},
        'va_consent': {'yes': True, 'no': False}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    
    return df

def get_analysis_datasets(data_dir="data", sample_size=None):
    """
    Load and return both real and fake datasets ready for analysis.
    
    Args:
        data_dir: Directory containing the data files
        sample_size: Number of rows to load from each file (None for all)
    
    Returns:
        tuple: (real_df, fake_df, combined_df)
    """
    # Get most recent files
    real_file = get_most_recent_file(data_dir, "real")
    fake_file = get_most_recent_file(data_dir, "fake")
    
    print(f"Loading real data: {Path(real_file).name}")
    print(f"Loading fake data: {Path(fake_file).name}")
    
    # Load raw data
    real_df = load_raw_data(real_file, sample_size)
    fake_df = load_raw_data(fake_file, sample_size)
    
    # Standardize
    real_clean = standardize_columns(real_df, 'real')
    fake_clean = standardize_columns(fake_df, 'fake')
    
    # Create combined dataset with common columns only
    common_cols = set(real_clean.columns).intersection(set(fake_clean.columns))
    combined_df = pd.concat([
        real_clean[list(common_cols)],
        fake_clean[list(common_cols)]
    ], ignore_index=True)
    
    print(f"Real data: {len(real_clean):,} rows, {len(real_clean.columns)} columns")
    print(f"Fake data: {len(fake_clean):,} rows, {len(fake_clean.columns)} columns")
    print(f"Combined: {len(combined_df):,} rows, {len(common_cols)} common columns")
    
    return real_clean, fake_clean, combined_df

def get_feature_summary(df, group_col='data_source'):
    """
    Generate summary statistics grouped by specified column.
    
    Args:
        df: DataFrame to summarize
        group_col: Column to group by
    
    Returns:
        dict: Summary statistics by group
    """
    summary = {}
    
    for group_name, group_df in df.groupby(group_col):
        summary[group_name] = {
            'count': len(group_df),
            'numeric_stats': group_df.select_dtypes(include=[np.number]).describe(),
            'categorical_counts': {
                col: group_df[col].value_counts().to_dict() 
                for col in group_df.select_dtypes(include=['object', 'bool']).columns
            },
            'missing_data': group_df.isnull().sum().to_dict()
        }
    
    return summary

# Analysis helper functions
def detect_outliers(df, columns, method='iqr', threshold=1.5):
    """Detect outliers in specified columns using IQR or Z-score method."""
    outliers = {}
    
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = df[z_scores > threshold].index.tolist()
    
    return outliers

def get_temporal_patterns(df, date_col='visit_date', group_col='data_source'):
    """Analyze temporal patterns in the data."""
    if date_col not in df.columns:
        return {}
    
    patterns = {}
    
    for group_name, group_df in df.groupby(group_col):
        group_df = group_df.copy()
        group_df['date'] = group_df[date_col].dt.date
        group_df['hour'] = group_df[date_col].dt.hour
        group_df['day_of_week'] = group_df[date_col].dt.day_name()
        
        patterns[group_name] = {
            'date_range': (group_df[date_col].min(), group_df[date_col].max()),
            'unique_dates': group_df['date'].nunique(),
            'submissions_per_date': group_df['date'].value_counts().describe(),
            'hourly_distribution': group_df['hour'].value_counts().sort_index(),
            'daily_distribution': group_df['day_of_week'].value_counts()
        }
    
    return patterns
