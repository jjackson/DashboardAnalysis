#!/usr/bin/env python3
"""
Feature extraction module - handles both visit-level and worker-level feature creation.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from tail_statistics import extract_universal_tail_features
from advanced_features import extract_advanced_worker_features
import re
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def extract_visit_features(real_df, fake_df):
    """
    Extract visit-level features (one row per visit).
    
    Args:
        real_df: Real visits dataframe
        fake_df: Fake visits dataframe
        
    Returns:
        tuple: (feature_matrix, labels, feature_names)
    """
    print("🔍 Extracting visit-level features...")
    
    # Combine datasets
    combined_df = pd.concat([real_df, fake_df], ignore_index=True)
    labels = np.concatenate([np.zeros(len(real_df)), np.ones(len(fake_df))])
    
    # Calculate population baselines from real data only
    baselines = _calculate_population_baselines(real_df)
    
    features = {}
    
    # 1. Household Size Z-Score (strongest signal from our analysis)
    if 'no_of_children' in combined_df.columns:
        hh_values = combined_df['no_of_children'].fillna(baselines['household_median'])
        hh_zscores = (hh_values - baselines['household_mean']) / (baselines['household_std'] + 1e-8)
        features['household_zscore'] = hh_zscores.values
        
        # Also add outlier detection
        q25, q75 = baselines['household_q25'], baselines['household_q75']
        iqr = q75 - q25
        outlier_scores = []
        for val in hh_values:
            if val > q75:
                outlier_scores.append((val - q75) / (iqr + 1e-8))
            elif val < q25:
                outlier_scores.append((q25 - val) / (iqr + 1e-8))
            else:
                outlier_scores.append(0)
        features['household_outlier'] = np.array(outlier_scores)
        
        # Upper tail features (large household indicators)
        features['large_household_indicator'] = (hh_values > 15).astype(int)  # >15 children
        features['medium_large_household_indicator'] = (hh_values > 10).astype(int)  # >10 children
        features['extreme_outlier_indicator'] = (hh_values > 1000).astype(int)  # Data quality errors
    
    # 2. MUAC Features (medical consistency)
    if 'soliciter_muac_cm' in combined_df.columns:
        muac_values = combined_df['soliciter_muac_cm'].fillna(baselines['muac_median'])
        muac_zscores = (muac_values - baselines['muac_mean']) / (baselines['muac_std'] + 1e-8)
        features['muac_zscore'] = muac_zscores.values
        
        # MUAC Percentiles
        real_muac = real_df['soliciter_muac_cm'].dropna()
        if len(real_muac) > 0:
            real_sorted = np.sort(real_muac.values)
            muac_percentiles = []
            for val in muac_values:
                if pd.isna(val):
                    muac_percentiles.append(0.5)  # Median percentile for missing
                else:
                    percentile = np.searchsorted(real_sorted, val) / len(real_sorted)
                    muac_percentiles.append(percentile)
            features['muac_percentile'] = np.array(muac_percentiles)
        
        # MUAC Outlier Detection
        q25, q75 = baselines.get('muac_q25', 12), baselines.get('muac_q75', 15)
        iqr = q75 - q25
        muac_outliers = []
        for val in muac_values:
            if pd.isna(val):
                muac_outliers.append(0)
            elif val > q75:
                muac_outliers.append((val - q75) / (iqr + 1e-8))
            elif val < q25:
                muac_outliers.append((q25 - val) / (iqr + 1e-8))
            else:
                muac_outliers.append(0)
        features['muac_outlier'] = np.array(muac_outliers)
    
    # 3. Age Features
    if 'childs_age_in_month' in combined_df.columns:
        age_values = combined_df['childs_age_in_month'].fillna(baselines['age_median'])
        age_zscores = (age_values - baselines['age_mean']) / (baselines['age_std'] + 1e-8)
        features['age_zscore'] = age_zscores.values
        
        # Age Percentiles
        real_age = real_df['childs_age_in_month'].dropna()
        if len(real_age) > 0:
            real_sorted = np.sort(real_age.values)
            age_percentiles = []
            for val in age_values:
                if pd.isna(val):
                    age_percentiles.append(0.5)  # Median percentile for missing
                else:
                    percentile = np.searchsorted(real_sorted, val) / len(real_sorted)
                    age_percentiles.append(percentile)
            features['age_percentile'] = np.array(age_percentiles)
        
        # Age Outlier Detection
        q25, q75 = baselines.get('age_q25', 12), baselines.get('age_q75', 36)
        iqr = q75 - q25
        age_outliers = []
        for val in age_values:
            if pd.isna(val):
                age_outliers.append(0)
            elif val > q75:
                age_outliers.append((val - q75) / (iqr + 1e-8))
            elif val < q25:
                age_outliers.append((q25 - val) / (iqr + 1e-8))
            else:
                age_outliers.append(0)
        features['age_outlier'] = np.array(age_outliers)
    
    # 4. Medical Consistency Scores
    medical_features = _calculate_medical_consistency(combined_df)
    features.update(medical_features)
    
    # 5. Categorical Distribution Features
    categorical_features = _calculate_categorical_features(combined_df, real_df, fake_df)
    features.update(categorical_features)
    
    # 6. Vaccination and Health Patterns
    pattern_features = _calculate_pattern_features(combined_df)
    features.update(pattern_features)
    
    # 7. Data Quality Indicators
    quality_features = _calculate_data_quality(combined_df)
    features.update(quality_features)
    
    # 8. Universal Tail Features for all numeric fields
    print("🔍 Calculating universal tail features...")
    numeric_fields = ['childs_age_in_month', 'no_of_children', 'soliciter_muac_cm']
    tail_features = extract_universal_tail_features(combined_df, real_df, fake_df, numeric_fields)
    features.update(tail_features)
    
    # Convert to matrix
    feature_names = list(features.keys())
    feature_matrix = np.column_stack([features[name] for name in feature_names])
    
    print(f"✅ Extracted {len(feature_names)} visit-level features from {len(combined_df):,} visits")
    for name in feature_names:
        print(f"   - {name}")
    
    return feature_matrix, labels, feature_names

def extract_worker_features(real_df, fake_df):
    """
    Extract worker-level features (one row per worker).
    
    Args:
        real_df: Real visits dataframe
        fake_df: Fake visits dataframe
        
    Returns:
        tuple: (feature_matrix, labels, feature_names)
    """
    print("👥 Extracting worker-level features...")
    
    # Calculate baselines from real workers only
    baselines = _calculate_worker_baselines(real_df)
    
    # Get all workers
    fake_worker_ids = set(fake_df['flw_id'].unique())
    real_worker_ids = set(real_df['flw_id'].unique())
    all_worker_ids = fake_worker_ids | real_worker_ids
    
    worker_features = {}
    worker_labels = []
    valid_workers = []
    
    for worker_id in all_worker_ids:
        # Get worker data
        if worker_id in fake_worker_ids:
            worker_data = fake_df[fake_df['flw_id'] == worker_id]
            label = 1  # Fake
        else:
            worker_data = real_df[real_df['flw_id'] == worker_id]
            label = 0  # Real
        
        # Skip workers with insufficient data
        if len(worker_data) < 2:
            continue
        
        # Calculate traditional worker features
        worker_stats = _calculate_worker_stats(worker_data, baselines)
        
        # Calculate advanced features (distribution, clustering, digit analysis)
        advanced_stats = extract_advanced_worker_features(worker_data, baselines)
        
        # Combine all features
        all_stats = {**worker_stats, **advanced_stats}
        
        # Store features
        for feature_name, value in all_stats.items():
            if feature_name not in worker_features:
                worker_features[feature_name] = []
            worker_features[feature_name].append(value)
        
        worker_labels.append(label)
        valid_workers.append(worker_id)
    
    # Ensure all workers have the same features (fix dimension mismatch)
    if worker_features:
        # Get all possible feature names from all workers
        all_feature_names = set()
        for worker_idx in range(len(valid_workers)):
            # Get features for this worker by reconstructing from stored lists
            worker_feature_names = set()
            for feature_name, feature_list in worker_features.items():
                if worker_idx < len(feature_list):
                    worker_feature_names.add(feature_name)
            all_feature_names.update(worker_feature_names)
        
        # Convert to sorted list for consistency
        feature_names = sorted(list(all_feature_names))
        
        # Ensure all workers have all features (fill missing with 0)
        for feature_name in feature_names:
            if feature_name not in worker_features:
                worker_features[feature_name] = [0] * len(valid_workers)
            else:
                # Pad with zeros if some workers are missing this feature
                while len(worker_features[feature_name]) < len(valid_workers):
                    worker_features[feature_name].append(0)
        
        # Convert to matrix
        feature_matrix = np.column_stack([worker_features[name] for name in feature_names])
        worker_labels = np.array(worker_labels)
    else:
        # No valid workers
        feature_names = []
        feature_matrix = np.array([]).reshape(0, 0)
        worker_labels = np.array([])
    
    fake_count = sum(worker_labels)
    real_count = len(worker_labels) - fake_count
    
    print(f"✅ Extracted {len(feature_names)} worker-level features from {len(valid_workers)} workers")
    print(f"   Workers: {real_count} real + {fake_count} fake")
    for name in feature_names:
        print(f"   - {name}")
    
    return feature_matrix, worker_labels, feature_names

def _calculate_population_baselines(real_df):
    """Calculate population statistics from real data."""
    baselines = {}
    
    # Household size statistics
    if 'no_of_children' in real_df.columns:
        hh_values = real_df['no_of_children'].dropna()
        baselines['household_mean'] = hh_values.mean()
        baselines['household_std'] = hh_values.std()
        baselines['household_median'] = hh_values.median()
        baselines['household_q25'] = hh_values.quantile(0.25)
        baselines['household_q75'] = hh_values.quantile(0.75)
    
    # MUAC statistics
    if 'soliciter_muac_cm' in real_df.columns:
        muac_values = real_df['soliciter_muac_cm'].dropna()
        baselines['muac_mean'] = muac_values.mean()
        baselines['muac_std'] = muac_values.std()
        baselines['muac_median'] = muac_values.median()
        baselines['muac_q25'] = muac_values.quantile(0.25)
        baselines['muac_q75'] = muac_values.quantile(0.75)
    
    # Age statistics
    if 'childs_age_in_month' in real_df.columns:
        age_values = real_df['childs_age_in_month'].dropna()
        baselines['age_mean'] = age_values.mean()
        baselines['age_std'] = age_values.std()
        baselines['age_median'] = age_values.median()
        baselines['age_q25'] = age_values.quantile(0.25)
        baselines['age_q75'] = age_values.quantile(0.75)
    
    return baselines

def _calculate_worker_baselines(real_df):
    """Calculate worker-level baselines from real workers."""
    baselines = {}
    
    # Name diversity baseline
    real_worker_name_stats = defaultdict(list)
    for _, row in real_df.iterrows():
        if pd.notna(row.get('child_name')) and row['child_name'] != '':
            real_worker_name_stats[row['flw_id']].append(str(row['child_name']).lower())
    
    name_diversities = []
    for worker_names in real_worker_name_stats.values():
        if len(worker_names) > 1:
            diversity = len(set(worker_names)) / len(worker_names)
            name_diversities.append(diversity)
    
    baselines['name_diversity_mean'] = np.mean(name_diversities) if name_diversities else 0.9
    
    # Phone quality baseline
    phone_qualities = []
    for _, row in real_df.iterrows():
        if pd.notna(row.get('household_phone')) and row['household_phone'] != '':
            phone_str = str(row['household_phone'])
            quality = _calculate_phone_quality(phone_str)
            phone_qualities.append(quality)
    
    baselines['phone_quality_mean'] = np.mean(phone_qualities) if phone_qualities else 0.8
    
    return baselines

def _calculate_worker_stats(worker_data, baselines):
    """Calculate comprehensive distributional statistics for a single worker across all fields."""
    from scipy import stats as scipy_stats
    
    worker_stats = {}
    
    # Define all fields to analyze (18 target fields)
    all_fields = [
        'childs_age_in_month', 'childs_gender', 'child_name', 'no_of_children',
        'soliciter_muac_cm', 'muac_colour', 'diagnosed_with_mal_past_3_months',
        'household_phone', 'childs_birth_certificate', 'childs_vaccination_card',
        'childs_health_card', 'child_recovered_from_malnutrition',
        'child_enrolled_in_school', 'child_attending_school',
        'child_reason_not_attending_school', 'child_grade_in_school',
        'child_repeated_grade', 'child_age_appropriate_grade'
    ]
    
    # 1. NUMERIC FIELDS - Full distributional statistics
    numeric_fields = ['childs_age_in_month', 'no_of_children', 'soliciter_muac_cm']
    
    for field in numeric_fields:
        if field in worker_data.columns:
            values = worker_data[field].dropna()
            if len(values) > 0:
                # Basic statistics
                worker_stats[f'{field}_mean'] = values.mean()
                worker_stats[f'{field}_std'] = values.std() if len(values) > 1 else 0
                worker_stats[f'{field}_min'] = values.min()
                worker_stats[f'{field}_max'] = values.max()
                worker_stats[f'{field}_median'] = values.median()
                
                # Percentiles
                worker_stats[f'{field}_p25'] = values.quantile(0.25)
                worker_stats[f'{field}_p75'] = values.quantile(0.75)
                worker_stats[f'{field}_p90'] = values.quantile(0.90)
                worker_stats[f'{field}_p95'] = values.quantile(0.95)
                
                # Distribution shape
                if len(values) > 2:
                    worker_stats[f'{field}_skewness'] = scipy_stats.skew(values)
                    worker_stats[f'{field}_kurtosis'] = scipy_stats.kurtosis(values)
                else:
                    worker_stats[f'{field}_skewness'] = 0
                    worker_stats[f'{field}_kurtosis'] = 0
                
                # Range and spread
                worker_stats[f'{field}_range'] = values.max() - values.min()
                worker_stats[f'{field}_iqr'] = values.quantile(0.75) - values.quantile(0.25)
                worker_stats[f'{field}_cv'] = values.std() / (values.mean() + 1e-8)  # Coefficient of variation
                
                # Outlier rates (using IQR method)
                q1, q3 = values.quantile(0.25), values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_rate = ((values < lower_bound) | (values > upper_bound)).mean()
                worker_stats[f'{field}_outlier_rate'] = outlier_rate
                
                # Comparison to baseline (real data)
                if field in baselines:
                    baseline_mean = baselines.get(f'{field}_mean', baselines.get('household_mean', 0))
                    baseline_std = baselines.get(f'{field}_std', baselines.get('household_std', 1))
                    
                    worker_stats[f'{field}_vs_baseline_mean'] = (values.mean() - baseline_mean) / (baseline_std + 1e-8)
                    worker_stats[f'{field}_vs_baseline_std'] = values.std() / (baseline_std + 1e-8)
            else:
                # Fill with zeros if no data
                for stat_suffix in ['_mean', '_std', '_min', '_max', '_median', '_p25', '_p75', '_p90', '_p95',
                                  '_skewness', '_kurtosis', '_range', '_iqr', '_cv', '_outlier_rate',
                                  '_vs_baseline_mean', '_vs_baseline_std']:
                    worker_stats[f'{field}{stat_suffix}'] = 0
    
    # 2. CATEGORICAL FIELDS - Diversity and pattern statistics
    categorical_fields = [
        'childs_gender', 'muac_colour', 'diagnosed_with_mal_past_3_months',
        'childs_birth_certificate', 'childs_vaccination_card', 'childs_health_card',
        'child_recovered_from_malnutrition', 'child_enrolled_in_school',
        'child_attending_school', 'child_reason_not_attending_school',
        'child_grade_in_school', 'child_repeated_grade', 'child_age_appropriate_grade'
    ]
    
    for field in categorical_fields:
        if field in worker_data.columns:
            values = worker_data[field].dropna()
            if len(values) > 0:
                # Diversity metrics
                unique_values = values.nunique()
                total_values = len(values)
                worker_stats[f'{field}_diversity'] = unique_values / total_values if total_values > 0 else 0
                worker_stats[f'{field}_unique_count'] = unique_values
                worker_stats[f'{field}_total_count'] = total_values
                
                # Most common value frequency
                most_common_freq = values.value_counts().iloc[0] / total_values if total_values > 0 else 0
                worker_stats[f'{field}_most_common_freq'] = most_common_freq
                
                # Missing rate
                total_rows = len(worker_data)
                missing_rate = (total_rows - total_values) / total_rows
                worker_stats[f'{field}_missing_rate'] = missing_rate
                
                # For binary fields, calculate yes/no ratios
                if field in ['childs_birth_certificate', 'childs_vaccination_card', 'childs_health_card',
                           'child_recovered_from_malnutrition', 'child_enrolled_in_school',
                           'child_attending_school', 'child_repeated_grade', 'child_age_appropriate_grade']:
                    yes_count = (values == 'yes').sum()
                    no_count = (values == 'no').sum()
                    total_binary = yes_count + no_count
                    if total_binary > 0:
                        worker_stats[f'{field}_yes_ratio'] = yes_count / total_binary
                        worker_stats[f'{field}_no_ratio'] = no_count / total_binary
                    else:
                        worker_stats[f'{field}_yes_ratio'] = 0
                        worker_stats[f'{field}_no_ratio'] = 0
            else:
                # Fill with zeros if no data
                for stat_suffix in ['_diversity', '_unique_count', '_total_count', '_most_common_freq', '_missing_rate']:
                    worker_stats[f'{field}{stat_suffix}'] = 0
                # Binary ratios
                if field in ['childs_birth_certificate', 'childs_vaccination_card', 'childs_health_card',
                           'child_recovered_from_malnutrition', 'child_enrolled_in_school',
                           'child_attending_school', 'child_repeated_grade', 'child_age_appropriate_grade']:
                    worker_stats[f'{field}_yes_ratio'] = 0
                    worker_stats[f'{field}_no_ratio'] = 0
    
    # 3. TEXT FIELDS - Special handling for names and phones
    if 'child_name' in worker_data.columns:
        names = [str(name).lower().strip() for name in worker_data['child_name'].dropna() if str(name).strip() != '']
        if len(names) > 0:
            # Name diversity
            unique_names = len(set(names))
            total_names = len(names)
            name_diversity = unique_names / total_names
            worker_stats['child_name_diversity'] = name_diversity
            worker_stats['child_name_unique_count'] = unique_names
            worker_stats['child_name_total_count'] = total_names
            worker_stats['child_name_repeat_rate'] = 1 - name_diversity
            
            # Name length statistics
            name_lengths = [len(name) for name in names]
            worker_stats['child_name_avg_length'] = np.mean(name_lengths)
            worker_stats['child_name_length_std'] = np.std(name_lengths) if len(name_lengths) > 1 else 0
            
            # Comparison to baseline
            baseline_diversity = baselines.get('name_diversity_mean', 0.8)
            worker_stats['child_name_vs_baseline_diversity'] = name_diversity - baseline_diversity
        else:
            for stat in ['child_name_diversity', 'child_name_unique_count', 'child_name_total_count',
                        'child_name_repeat_rate', 'child_name_avg_length', 'child_name_length_std',
                        'child_name_vs_baseline_diversity']:
                worker_stats[stat] = 0
    
    if 'household_phone' in worker_data.columns:
        phones = [str(phone) for phone in worker_data['household_phone'].dropna() if str(phone).strip() != '']
        if len(phones) > 0:
            # Phone diversity
            unique_phones = len(set(phones))
            total_phones = len(phones)
            phone_diversity = unique_phones / total_phones
            worker_stats['household_phone_diversity'] = phone_diversity
            worker_stats['household_phone_unique_count'] = unique_phones
            worker_stats['household_phone_repeat_rate'] = 1 - phone_diversity
            
            # Phone quality
            qualities = [_calculate_phone_quality(phone) for phone in phones]
            worker_stats['household_phone_quality_rate'] = np.mean(qualities)
            
            # Comparison to baseline
            baseline_quality = baselines.get('phone_quality_mean', 0.5)
            worker_stats['household_phone_vs_baseline_quality'] = np.mean(qualities) - baseline_quality
        else:
            for stat in ['household_phone_diversity', 'household_phone_unique_count', 'household_phone_repeat_rate',
                        'household_phone_quality_rate', 'household_phone_vs_baseline_quality']:
                worker_stats[stat] = 0
    
    # 4. CROSS-FIELD CONSISTENCY PATTERNS
    # Medical consistency (MUAC vs malnutrition diagnosis)
    consistency_scores = []
    for _, row in worker_data.iterrows():
        if pd.notna(row.get('diagnosed_with_mal_past_3_months')) and pd.notna(row.get('muac_colour')):
            diagnosed = row['diagnosed_with_mal_past_3_months']
            muac_color = row['muac_colour']
            
            if diagnosed == 'yes' and muac_color in ['red', 'yellow', 'orange']:
                consistency_scores.append(1)
            elif diagnosed == 'no' and muac_color == 'green':
                consistency_scores.append(1)
            else:
                consistency_scores.append(0)
    
    worker_stats['medical_consistency_rate'] = np.mean(consistency_scores) if consistency_scores else 0.5
    
    # Age-grade consistency
    age_grade_consistency = []
    for _, row in worker_data.iterrows():
        if pd.notna(row.get('childs_age_in_month')) and pd.notna(row.get('child_grade_in_school')):
            age_months = row['childs_age_in_month']
            grade = str(row['child_grade_in_school']).lower()
            
            # Simple age-grade consistency check
            expected_age_years = age_months / 12
            if 'nursery' in grade or 'pre' in grade:
                expected_grade_age = 4
            elif any(num in grade for num in ['1', 'one']):
                expected_grade_age = 6
            elif any(num in grade for num in ['2', 'two']):
                expected_grade_age = 7
            elif any(num in grade for num in ['3', 'three']):
                expected_grade_age = 8
            else:
                expected_grade_age = 7  # Default
            
            age_diff = abs(expected_age_years - expected_grade_age)
            age_grade_consistency.append(1 if age_diff <= 2 else 0)  # Within 2 years is consistent
    
    worker_stats['age_grade_consistency_rate'] = np.mean(age_grade_consistency) if age_grade_consistency else 0.5
    
    # 5. OVERALL DATA QUALITY METRICS
    all_target_fields = ['childs_age_in_month', 'childs_gender', 'child_name', 'no_of_children',
                        'soliciter_muac_cm', 'muac_colour', 'diagnosed_with_mal_past_3_months']
    
    completeness_scores = []
    for _, row in worker_data.iterrows():
        complete_fields = sum(1 for field in all_target_fields 
                            if pd.notna(row.get(field)) and str(row.get(field)).strip() != '')
        completeness_scores.append(complete_fields / len(all_target_fields))
    
    worker_stats['overall_data_completeness'] = np.mean(completeness_scores)
    worker_stats['total_visits'] = len(worker_data)
    
    return worker_stats

def _calculate_phone_quality(phone_str):
    """Calculate quality score for a phone number."""
    has_sequential = bool(re.search(r'(012|123|234|345|456|567|678|789)', phone_str))
    has_repeated = bool(re.search(r'(\d)\1{2,}', phone_str))
    valid_length = 10 <= len(re.sub(r'\D', '', phone_str)) <= 11
    
    return valid_length and not has_sequential and not has_repeated

def _calculate_medical_consistency(combined_df):
    """Calculate medical consistency scores per visit."""
    features = {}
    
    # MUAC-malnutrition consistency
    muac_consistency = []
    for _, row in combined_df.iterrows():
        if pd.notna(row.get('diagnosed_with_mal_past_3_months')) and pd.notna(row.get('muac_colour')):
            diagnosed = row['diagnosed_with_mal_past_3_months']
            muac_color = row['muac_colour']
            
            if diagnosed == 'yes' and muac_color in ['red', 'yellow', 'orange']:
                muac_consistency.append(1)  # Consistent
            elif diagnosed == 'no' and muac_color == 'green':
                muac_consistency.append(1)  # Consistent
            elif diagnosed == 'yes' and muac_color == 'green':
                muac_consistency.append(-1)  # Inconsistent
            elif diagnosed == 'no' and muac_color in ['red', 'yellow', 'orange']:
                muac_consistency.append(-1)  # Inconsistent
            else:
                muac_consistency.append(0)  # Unclear
        else:
            muac_consistency.append(0)  # Missing data
    
    features['muac_malnutrition_consistency'] = np.array(muac_consistency)
    
    # Age-MUAC consistency
    age_muac_consistency = []
    for _, row in combined_df.iterrows():
        age = row.get('childs_age_in_month')
        muac = row.get('soliciter_muac_cm')
        
        if pd.notna(age) and pd.notna(muac):
            if age < 6 and muac > 0:
                age_muac_consistency.append(-1)  # Too young for MUAC
            elif 6 <= age <= 59:
                age_muac_consistency.append(1)  # Appropriate age
            else:
                age_muac_consistency.append(0)  # Edge case
        else:
            age_muac_consistency.append(0)  # Missing data
    
    features['age_muac_consistency'] = np.array(age_muac_consistency)
    
    return features

def _calculate_categorical_features(combined_df, real_df, fake_df):
    """Calculate categorical distribution features."""
    features = {}
    
    # Gender distribution analysis
    if 'childs_gender' in combined_df.columns:
        real_gender_counts = real_df['childs_gender'].value_counts(normalize=True)
        fake_gender_counts = fake_df['childs_gender'].value_counts(normalize=True)
        
        gender_features = []
        for value in combined_df['childs_gender']:
            if pd.isna(value) or value == '':
                gender_features.append([0, 0])
            else:
                real_freq = real_gender_counts.get(value, 0)
                fake_freq = fake_gender_counts.get(value, 0)
                gender_features.append([real_freq, fake_freq])
        
        gender_array = np.array(gender_features)
        features['gender_real_freq'] = gender_array[:, 0]
        features['gender_fake_freq'] = gender_array[:, 1]
    
    # MUAC color distribution analysis
    if 'muac_colour' in combined_df.columns:
        real_color_counts = real_df['muac_colour'].value_counts(normalize=True)
        fake_color_counts = fake_df['muac_colour'].value_counts(normalize=True)
        
        color_features = []
        for value in combined_df['muac_colour']:
            if pd.isna(value) or value == '':
                color_features.append([0, 0])
            else:
                real_freq = real_color_counts.get(value, 0)
                fake_freq = fake_color_counts.get(value, 0)
                color_features.append([real_freq, fake_freq])
        
        color_array = np.array(color_features)
        features['muac_color_real_freq'] = color_array[:, 0]
        features['muac_color_fake_freq'] = color_array[:, 1]
    
    return features

def _calculate_pattern_features(combined_df):
    """Calculate vaccination and health pattern features."""
    features = {}
    
    # Vaccination pattern analysis
    vaccination_fields = ['received_va_dose_before', 'received_any_vaccine', 'recent_va_dose']
    available_vac_fields = [f for f in vaccination_fields if f in combined_df.columns]
    
    if available_vac_fields:
        vac_patterns = []
        for _, row in combined_df.iterrows():
            yes_count = sum(1 for field in available_vac_fields if row.get(field) == 'yes')
            no_count = sum(1 for field in available_vac_fields if row.get(field) == 'no')
            missing_count = sum(1 for field in available_vac_fields 
                              if pd.isna(row.get(field)) or row.get(field) == '')
            
            total_fields = len(available_vac_fields)
            yes_ratio = yes_count / total_fields if total_fields > 0 else 0
            no_ratio = no_count / total_fields if total_fields > 0 else 0
            missing_ratio = missing_count / total_fields if total_fields > 0 else 0
            
            vac_patterns.append([yes_ratio, no_ratio, missing_ratio])
        
        vac_array = np.array(vac_patterns)
        features['vaccination_yes_ratio'] = vac_array[:, 0]
        features['vaccination_no_ratio'] = vac_array[:, 1]
        features['vaccination_missing_ratio'] = vac_array[:, 2]
    
    # Health status pattern analysis
    health_fields = ['have_glasses', 'va_child_unwell_today', 'diarrhea_last_month']
    available_health_fields = [f for f in health_fields if f in combined_df.columns]
    
    if available_health_fields:
        health_patterns = []
        for _, row in combined_df.iterrows():
            yes_count = sum(1 for field in available_health_fields if row.get(field) == 'yes')
            missing_count = sum(1 for field in available_health_fields 
                              if pd.isna(row.get(field)) or row.get(field) == '')
            
            total_fields = len(available_health_fields)
            yes_ratio = yes_count / total_fields if total_fields > 0 else 0
            missing_ratio = missing_count / total_fields if total_fields > 0 else 0
            
            health_patterns.append([yes_ratio, missing_ratio])
        
        health_array = np.array(health_patterns)
        features['health_yes_ratio'] = health_array[:, 0]
        features['health_missing_ratio'] = health_array[:, 1]
    
    # Recovery pattern analysis
    if 'diarrhea_last_month' in combined_df.columns and 'did_the_child_recover' in combined_df.columns:
        recovery_patterns = []
        for _, row in combined_df.iterrows():
            diarrhea = row.get('diarrhea_last_month')
            recovered = row.get('did_the_child_recover')
            
            # Logical consistency: if no diarrhea, recovery question shouldn't apply
            if pd.isna(diarrhea) or pd.isna(recovered):
                consistency_score = 0
            elif diarrhea == 'no' and recovered == 'yes':
                consistency_score = -1  # Inconsistent: no diarrhea but recovered
            elif diarrhea == 'yes':
                consistency_score = 1 if recovered == 'yes' else 0.5  # Had diarrhea, recovery status
            else:
                consistency_score = 0
            
            recovery_patterns.append(consistency_score)
        
        features['recovery_consistency'] = np.array(recovery_patterns)
    
    return features

def _calculate_data_quality(combined_df):
    """Calculate data quality indicators per visit."""
    features = {}
    
    # Missing data rate
    important_fields = ['childs_age_in_month', 'childs_gender', 'child_name', 'no_of_children',
                       'soliciter_muac_cm', 'muac_colour', 'diagnosed_with_mal_past_3_months']
    
    missing_rates = []
    for _, row in combined_df.iterrows():
        missing = sum(1 for field in important_fields 
                     if pd.isna(row.get(field)) or row.get(field) == '')
        missing_rates.append(missing / len(important_fields))
    
    features['missing_data_rate'] = np.array(missing_rates)
    
    return features

