#!/usr/bin/env python3
"""
Tail statistics module - calculates tail-specific statistics for numeric fields.
These can be applied universally to detect distribution differences.
"""

import numpy as np
import pandas as pd
from scipy import stats

def calculate_tail_statistics(values, percentile_thresholds=[90, 95, 99]):
    """
    Calculate comprehensive tail statistics for a numeric field.
    
    Args:
        values: Array of numeric values
        percentile_thresholds: Percentiles to use for tail analysis
        
    Returns:
        dict: Dictionary of tail statistics
    """
    values_clean = np.array(values)
    values_clean = values_clean[~np.isnan(values_clean)]
    
    if len(values_clean) == 0:
        return {}
    
    tail_stats = {}
    
    # 1. Skewness (measures tail asymmetry)
    tail_stats['skewness'] = stats.skew(values_clean)
    
    # 2. Kurtosis (measures tail heaviness)
    tail_stats['kurtosis'] = stats.kurtosis(values_clean)
    
    # 3. Percentile ratios (tail spread measures)
    p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(values_clean, [5, 10, 25, 50, 75, 90, 95, 99])
    
    # Upper tail spread
    tail_stats['upper_tail_ratio'] = (p99 - p95) / (p95 - p90 + 1e-8)  # How much the extreme upper tail spreads
    tail_stats['upper_mid_ratio'] = (p95 - p90) / (p90 - p75 + 1e-8)   # Upper-middle tail spread
    
    # Lower tail spread  
    tail_stats['lower_tail_ratio'] = (p10 - p5) / (p25 - p10 + 1e-8) if len(values_clean) > 20 else 0
    tail_stats['lower_mid_ratio'] = (p25 - p10) / (p50 - p25 + 1e-8)   # Lower-middle tail spread
    
    # 4. Tail density measures
    # What percentage of data is in extreme tails?
    total_range = p99 - p10 + 1e-8
    upper_tail_density = np.sum(values_clean > p95) / len(values_clean)
    lower_tail_density = np.sum(values_clean < p10) / len(values_clean)
    
    tail_stats['upper_tail_density'] = upper_tail_density
    tail_stats['lower_tail_density'] = lower_tail_density
    
    # 5. Tail deviation from normal
    # How much do the tails deviate from what we'd expect in a normal distribution?
    try:
        # Fit normal distribution to the data
        mu, sigma = stats.norm.fit(values_clean)
        
        # Expected percentiles under normal distribution
        expected_p95 = stats.norm.ppf(0.95, mu, sigma)
        expected_p99 = stats.norm.ppf(0.99, mu, sigma)
        expected_p10 = stats.norm.ppf(0.10, mu, sigma)
        expected_p5 = stats.norm.ppf(0.05, mu, sigma)
        
        # Tail deviation ratios
        tail_stats['upper_tail_deviation'] = (p99 - expected_p99) / (sigma + 1e-8)
        tail_stats['upper_mid_deviation'] = (p95 - expected_p95) / (sigma + 1e-8)
        tail_stats['lower_tail_deviation'] = (expected_p5 - p5) / (sigma + 1e-8) if len(values_clean) > 20 else 0
        tail_stats['lower_mid_deviation'] = (expected_p10 - p10) / (sigma + 1e-8)
        
    except:
        # If normal fitting fails, set to 0
        tail_stats['upper_tail_deviation'] = 0
        tail_stats['upper_mid_deviation'] = 0
        tail_stats['lower_tail_deviation'] = 0
        tail_stats['lower_mid_deviation'] = 0
    
    # 6. Interquartile range vs tail range ratio
    iqr = p75 - p25
    tail_range = p95 - p10
    tail_stats['tail_to_iqr_ratio'] = tail_range / (iqr + 1e-8)
    
    # 7. Extreme outlier indicators
    # Using 1.5 * IQR rule and 3-sigma rule
    iqr_lower = p25 - 1.5 * iqr
    iqr_upper = p75 + 1.5 * iqr
    sigma_3_lower = mu - 3 * sigma if 'mu' in locals() else p50 - 3 * np.std(values_clean)
    sigma_3_upper = mu + 3 * sigma if 'mu' in locals() else p50 + 3 * np.std(values_clean)
    
    tail_stats['iqr_outlier_rate'] = (np.sum(values_clean < iqr_lower) + np.sum(values_clean > iqr_upper)) / len(values_clean)
    tail_stats['sigma3_outlier_rate'] = (np.sum(values_clean < sigma_3_lower) + np.sum(values_clean > sigma_3_upper)) / len(values_clean)
    
    return tail_stats

def calculate_comparative_tail_features(real_values, fake_values, field_name):
    """
    Calculate comparative tail statistics between real and fake data for a field.
    
    Args:
        real_values: Real data values
        fake_values: Fake data values  
        field_name: Name of the field
        
    Returns:
        dict: Comparative tail features
    """
    real_stats = calculate_tail_statistics(real_values)
    fake_stats = calculate_tail_statistics(fake_values)
    
    if not real_stats or not fake_stats:
        return {}
    
    comparative_features = {}
    
    # Calculate differences for each tail statistic
    for stat_name in real_stats.keys():
        real_val = real_stats[stat_name]
        fake_val = fake_stats[stat_name]
        
        # Difference
        diff = fake_val - real_val
        comparative_features[f'{field_name}_tail_{stat_name}_diff'] = diff
        
        # Ratio (for non-zero values)
        if abs(real_val) > 1e-8:
            ratio = fake_val / real_val
            comparative_features[f'{field_name}_tail_{stat_name}_ratio'] = ratio
    
    return comparative_features

def extract_universal_tail_features(combined_df, real_df, fake_df, numeric_fields=None):
    """
    DEPRECATED: Tail features not adding significant value.
    Keeping function for backward compatibility but returning empty dict.
    """
    print("⚠️  Tail features disabled - not adding significant detection value")
    return {}

def extract_universal_tail_features_DISABLED(combined_df, real_df, fake_df, numeric_fields=None):
    """
    Extract tail-based features for all numeric fields.
    
    Args:
        combined_df: Combined dataset
        real_df: Real data only
        fake_df: Fake data only
        numeric_fields: List of numeric fields to analyze (if None, auto-detect)
        
    Returns:
        dict: Dictionary of tail features for each visit
    """
    if numeric_fields is None:
        # Auto-detect numeric fields
        numeric_fields = []
        for col in combined_df.columns:
            if combined_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Skip ID fields and other non-meaningful numeric fields
                if not any(skip in col.lower() for skip in ['id', 'uuid', 'timestamp', 'date']):
                    numeric_fields.append(col)
    
    print(f"🔍 Calculating tail features for {len(numeric_fields)} numeric fields:")
    for field in numeric_fields:
        print(f"   - {field}")
    
    tail_features = {}
    
    for field in numeric_fields:
        if field not in combined_df.columns:
            continue
            
        print(f"📊 Processing tail features for: {field}")
        
        # Get clean values
        real_values = real_df[field].dropna()
        fake_values = fake_df[field].dropna()
        combined_values = combined_df[field].fillna(combined_df[field].median())
        
        if len(real_values) == 0 or len(fake_values) == 0:
            continue
        
        # Calculate baseline tail statistics from real data
        real_tail_stats = calculate_tail_statistics(real_values)
        
        if not real_tail_stats:
            continue
        
        # OPTIMIZED: Pre-calculate all percentiles ONCE instead of per-visit
        real_vals_array = np.array(real_values)
        real_vals_sorted = np.sort(real_vals_array)
        real_std = np.std(real_vals_array)
        p1_real = np.percentile(real_vals_array, 1)
        p5_real = np.percentile(real_vals_array, 5)
        p95_real = np.percentile(real_vals_array, 95)
        p99_real = np.percentile(real_vals_array, 99)
        n_real = len(real_vals_array)
        
        # OPTIMIZED: Vectorized calculations for ALL visits at once
        combined_vals_array = np.array(combined_values)
        
        # Calculate percentile positions for all values at once
        percentile_positions = np.searchsorted(real_vals_sorted, combined_vals_array) / n_real
        
        # Vectorized tail indicators
        upper_tail_indicators = (percentile_positions > 0.95).astype(int)
        lower_tail_indicators = (percentile_positions < 0.05).astype(int)
        extreme_upper_indicators = (percentile_positions > 0.99).astype(int)
        extreme_lower_indicators = (percentile_positions < 0.01).astype(int)
        
        # Vectorized distance calculations
        upper_tail_distances = np.maximum(0, combined_vals_array - p95_real) / (p99_real - p95_real + 1e-8)
        extreme_upper_distances = np.maximum(0, combined_vals_array - p99_real) / (real_std + 1e-8)
        lower_tail_distances = np.maximum(0, p5_real - combined_vals_array) / (p5_real - p1_real + 1e-8)
        extreme_lower_distances = np.maximum(0, p1_real - combined_vals_array) / (real_std + 1e-8)
        
        # Add to features dictionary
        tail_features[f'{field}_tail_percentile_position'] = percentile_positions
        tail_features[f'{field}_tail_upper_tail_indicator'] = upper_tail_indicators
        tail_features[f'{field}_tail_lower_tail_indicator'] = lower_tail_indicators
        tail_features[f'{field}_tail_extreme_upper_indicator'] = extreme_upper_indicators
        tail_features[f'{field}_tail_extreme_lower_indicator'] = extreme_lower_indicators
        tail_features[f'{field}_tail_upper_tail_distance'] = upper_tail_distances
        tail_features[f'{field}_tail_extreme_upper_distance'] = extreme_upper_distances
        tail_features[f'{field}_tail_lower_tail_distance'] = lower_tail_distances
        tail_features[f'{field}_tail_extreme_lower_distance'] = extreme_lower_distances
        
        print(f"   ✅ Generated 9 tail features for {field} in vectorized mode")
    
    return tail_features

# Test the tail statistics
if __name__ == "__main__":
    # Test with some sample data
    normal_data = np.random.normal(10, 2, 1000)
    skewed_data = np.random.exponential(2, 1000)
    
    print("Normal data tail stats:")
    normal_stats = calculate_tail_statistics(normal_data)
    for k, v in normal_stats.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nSkewed data tail stats:")
    skewed_stats = calculate_tail_statistics(skewed_data)
    for k, v in skewed_stats.items():
        print(f"  {k}: {v:.4f}")
