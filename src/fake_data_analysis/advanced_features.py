#!/usr/bin/env python3
"""
Advanced feature extraction for fake data detection.
Focuses on distribution shape, clustering, and digit bias analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

def extract_distribution_features(worker_data, field_name, population_baseline=None):
    """
    Extract distribution shape and normality features for a numeric field.
    
    Args:
        worker_data: Series of numeric values for this worker
        field_name: Name of the field being analyzed
        population_baseline: Population distribution for comparison (optional)
    
    Returns:
        dict: Distribution-based features
    """
    features = {}
    
    if len(worker_data) < 3:
        # Not enough data for meaningful distribution analysis
        return {f'{field_name}_dist_{key}': 0 for key in [
            'normality_pvalue', 'is_normal', 'skewness_abs', 'kurtosis_abs',
            'num_modes', 'uniform_fit_quality', 'ks_vs_population'
        ]}
    
    values = np.array(worker_data.dropna())
    
    # 1. Normality Testing
    try:
        # Shapiro-Wilk test (good for small samples)
        stat, p_value = stats.shapiro(values)
        features[f'{field_name}_dist_normality_pvalue'] = p_value
        features[f'{field_name}_dist_is_normal'] = 1 if p_value > 0.05 else 0
    except:
        features[f'{field_name}_dist_normality_pvalue'] = 0
        features[f'{field_name}_dist_is_normal'] = 0
    
    # 2. Distribution Shape
    features[f'{field_name}_dist_skewness_abs'] = abs(stats.skew(values))
    features[f'{field_name}_dist_kurtosis_abs'] = abs(stats.kurtosis(values))
    
    # 3. Multimodality Detection (simple peak counting)
    try:
        hist, bin_edges = np.histogram(values, bins=min(10, len(np.unique(values))))
        # Count local maxima in histogram
        peaks = 0
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks += 1
        features[f'{field_name}_dist_num_modes'] = peaks
    except:
        features[f'{field_name}_dist_num_modes'] = 0
    
    # 4. Simple Uniformity Check (fast approximation)
    try:
        # Check if values are roughly evenly spaced (simpler than KS test)
        if len(np.unique(values)) > 2:
            sorted_vals = np.sort(values)
            gaps = np.diff(sorted_vals)
            gap_uniformity = 1 - (np.std(gaps) / (np.mean(gaps) + 1e-6)) if np.mean(gaps) > 0 else 0
            features[f'{field_name}_dist_uniform_fit_quality'] = max(0, gap_uniformity)
        else:
            features[f'{field_name}_dist_uniform_fit_quality'] = 0
    except:
        features[f'{field_name}_dist_uniform_fit_quality'] = 0
    
    # 5. Fast Population Comparison (avoid expensive KS test)
    if population_baseline is not None and len(population_baseline) > 10:
        try:
            # Simple percentile comparison (much faster than KS test)
            worker_percentiles = np.percentile(values, [25, 50, 75])
            pop_percentiles = np.percentile(population_baseline, [25, 50, 75])
            
            # Calculate similarity based on percentile differences
            percentile_diffs = np.abs(worker_percentiles - pop_percentiles)
            pop_iqr = np.percentile(population_baseline, 75) - np.percentile(population_baseline, 25)
            normalized_diffs = percentile_diffs / (pop_iqr + 1e-6)
            similarity = np.exp(-np.mean(normalized_diffs))  # Exponential decay similarity
            
            features[f'{field_name}_dist_ks_vs_population'] = similarity
        except:
            features[f'{field_name}_dist_ks_vs_population'] = 0.5
    else:
        features[f'{field_name}_dist_ks_vs_population'] = 0.5
    
    return features

def extract_clustering_features(worker_data, field_name, population_data=None):
    """
    FAST clustering-based features using simple statistical measures.
    Focuses on detecting artificial patterns without expensive ML operations.
    
    Args:
        worker_data: Series of numeric values for this worker
        field_name: Name of the field being analyzed
        population_data: Population data for comparison (optional)
    
    Returns:
        dict: Fast clustering-based features
    """
    features = {}
    
    if len(worker_data) < 3:
        # Not enough data
        return {f'{field_name}_cluster_{key}': 0 for key in [
            'value_concentration', 'gap_pattern', 'population_deviation'
        ]}
    
    values = np.array(worker_data.dropna())
    
    # 1. Value Concentration (how clustered are the values?)
    # Simple measure: ratio of unique values to total values
    unique_ratio = len(np.unique(values)) / len(values)
    features[f'{field_name}_cluster_value_concentration'] = 1 - unique_ratio  # Higher = more concentrated
    
    # 2. Gap Pattern Detection (are there suspicious gaps/clusters?)
    if len(values) > 3:
        sorted_vals = np.sort(values)
        gaps = np.diff(sorted_vals)
        if len(gaps) > 0 and np.std(gaps) > 0:
            # Coefficient of variation of gaps (high = irregular spacing)
            gap_cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
            features[f'{field_name}_cluster_gap_pattern'] = gap_cv
        else:
            features[f'{field_name}_cluster_gap_pattern'] = 0
    else:
        features[f'{field_name}_cluster_gap_pattern'] = 0
    
    # 3. Population Deviation (fast comparison)
    if population_data is not None and len(population_data) > 10:
        try:
            pop_values = np.array(population_data)
            # Simple statistical comparison (much faster than clustering)
            worker_mean = np.mean(values)
            worker_std = np.std(values)
            pop_mean = np.mean(pop_values)
            pop_std = np.std(pop_values)
            
            # Normalized difference in means and stds
            mean_diff = abs(worker_mean - pop_mean) / (pop_std + 1e-6)
            std_ratio = worker_std / (pop_std + 1e-6) if pop_std > 0 else 1
            
            # Combined deviation score
            deviation = mean_diff + abs(np.log(std_ratio + 1e-6))
            features[f'{field_name}_cluster_population_deviation'] = min(deviation, 10)  # Cap at 10
            
        except:
            features[f'{field_name}_cluster_population_deviation'] = 0
    else:
        features[f'{field_name}_cluster_population_deviation'] = 0
    
    return features

def extract_muac_digit_features(muac_values):
    """
    Extract digit bias and precision features from MUAC measurements.
    Focuses on detecting fabricated vs. real measurement patterns.
    
    Args:
        muac_values: Series of MUAC measurements (floats like 12.3, 14.7)
    
    Returns:
        dict: Digit bias features
    """
    features = {}
    
    if len(muac_values) < 3:
        # Not enough data
        return {f'muac_digit_{key}': 0 for key in [
            'last_digit_entropy', 'round_number_rate', 'decimal_5_bias', 
            'decimal_0_bias', 'precision_consistency', 'digit_repetition_rate'
        ]}
    
    values = muac_values.dropna()
    
    # 1. Last Digit Analysis (decimal place)
    decimal_digits = []
    for val in values:
        # Extract decimal digit (e.g., 12.3 -> 3, 14.0 -> 0)
        decimal_part = val - int(val)
        decimal_digit = int(round(decimal_part * 10)) % 10
        decimal_digits.append(decimal_digit)
    
    # Calculate entropy of decimal digits (uniform = high entropy, biased = low entropy)
    digit_counts = Counter(decimal_digits)
    total = len(decimal_digits)
    entropy = -sum((count/total) * np.log2(count/total) for count in digit_counts.values() if count > 0)
    max_entropy = np.log2(10)  # Maximum possible entropy for 10 digits
    features['muac_digit_last_digit_entropy'] = entropy / max_entropy
    
    # 2. Round Number Bias
    round_numbers = sum(1 for d in decimal_digits if d in [0, 5])  # .0 and .5 are "round"
    features['muac_digit_round_number_rate'] = round_numbers / total
    
    # 3. Specific Digit Biases
    features['muac_digit_decimal_5_bias'] = decimal_digits.count(5) / total
    features['muac_digit_decimal_0_bias'] = decimal_digits.count(0) / total
    
    # 4. Precision Consistency
    # Check if measurements always use same decimal precision
    decimal_places = []
    for val in values:
        # Count decimal places
        str_val = f"{val:.10f}".rstrip('0')
        if '.' in str_val:
            decimal_places.append(len(str_val.split('.')[1]))
        else:
            decimal_places.append(0)
    
    precision_consistency = len(set(decimal_places)) / len(decimal_places) if decimal_places else 0
    features['muac_digit_precision_consistency'] = 1 - precision_consistency  # Higher = more consistent
    
    # 5. Digit Repetition Pattern Detection
    # Look for suspicious patterns like 12.2, 13.3, 14.4
    repetition_count = 0
    for val in values:
        str_val = f"{val:.1f}"
        if '.' in str_val:
            integer_part = str_val.split('.')[0]
            decimal_part = str_val.split('.')[1]
            # Check if last digit of integer matches decimal digit
            if len(integer_part) > 0 and len(decimal_part) > 0:
                if integer_part[-1] == decimal_part[0]:
                    repetition_count += 1
    
    features['muac_digit_repetition_rate'] = repetition_count / total
    
    return features

def extract_advanced_worker_features(worker_data, population_baselines=None):
    """
    Extract all advanced features for a single worker.
    
    Args:
        worker_data: DataFrame of worker's visits
        population_baselines: Dict of population data for comparison
    
    Returns:
        dict: All advanced features
    """
    features = {}
    
    # Numeric fields for distribution and clustering analysis
    numeric_fields = ['childs_age_in_month', 'no_of_children', 'soliciter_muac_cm']
    
    for field in numeric_fields:
        if field in worker_data.columns:
            field_data = worker_data[field].dropna()
            
            if len(field_data) > 0:
                # Distribution features
                pop_baseline = population_baselines.get(field) if population_baselines else None
                dist_features = extract_distribution_features(field_data, field, pop_baseline)
                features.update(dist_features)
                
                # Clustering features
                cluster_features = extract_clustering_features(field_data, field, pop_baseline)
                features.update(cluster_features)
    
    # Special MUAC digit analysis
    if 'soliciter_muac_cm' in worker_data.columns:
        muac_data = worker_data['soliciter_muac_cm'].dropna()
        if len(muac_data) > 0:
            digit_features = extract_muac_digit_features(muac_data)
            features.update(digit_features)
    
    return features

# Test function
if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    
    # Simulate real worker data (more natural)
    real_ages = np.random.normal(30, 8, 20).clip(6, 59)
    real_muac = np.random.normal(14.2, 1.5, 20).clip(10, 18)
    
    # Simulate fake worker data (more biased)
    fake_ages = [24, 36, 24, 36, 30, 30, 24, 36]  # Repetitive
    fake_muac = [12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5]  # Too regular
    
    print("Testing advanced features...")
    
    # Test distribution features
    real_features = extract_distribution_features(pd.Series(real_ages), 'age')
    fake_features = extract_distribution_features(pd.Series(fake_ages), 'age')
    
    print(f"Real worker normality p-value: {real_features.get('age_dist_normality_pvalue', 0):.3f}")
    print(f"Fake worker normality p-value: {fake_features.get('age_dist_normality_pvalue', 0):.3f}")
    
    # Test MUAC digit features
    real_muac_features = extract_muac_digit_features(pd.Series(real_muac))
    fake_muac_features = extract_muac_digit_features(pd.Series(fake_muac))
    
    print(f"Real MUAC digit entropy: {real_muac_features.get('muac_digit_last_digit_entropy', 0):.3f}")
    print(f"Fake MUAC digit entropy: {fake_muac_features.get('muac_digit_last_digit_entropy', 0):.3f}")
    print(f"Fake MUAC round number rate: {fake_muac_features.get('muac_digit_round_number_rate', 0):.3f}")
