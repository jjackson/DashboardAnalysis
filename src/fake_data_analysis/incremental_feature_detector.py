"""
Incremental Feature Selection for Fake Data Detection
Builds features one by one, showing the value added by each feature.
Stops when additional features don't provide significant improvement.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from data_utils import get_analysis_datasets

class IncrementalFeatureDetector:
    """
    Builds features incrementally, showing value of each addition.
    """
    
    def __init__(self, min_improvement=0.01):
        """
        Args:
            min_improvement: Minimum AUC improvement to justify adding a feature
        """
        self.min_improvement = min_improvement
        self.feature_history = []
        self.performance_history = []
        
    def extract_single_feature_type(self, real_df, fake_df, feature_type):
        """
        Extract a specific type of feature for analysis.
        
        Args:
            feature_type: 'muac_basic', 'muac_advanced', 'age_basic', 'age_advanced', 'household_basic'
        """
        print(f"Extracting {feature_type} features...")
        
        # Combine datasets
        combined_df = pd.concat([real_df, fake_df], ignore_index=True)
        labels = np.concatenate([np.zeros(len(real_df)), np.ones(len(fake_df))])
        
        if feature_type == 'muac_basic':
            # Basic MUAC z-score from real population
            if 'soliciter_muac_cm' in combined_df.columns:
                real_muac = real_df['soliciter_muac_cm'].dropna()
                if len(real_muac) > 0:
                    real_mean = real_muac.mean()
                    real_std = real_muac.std()
                    
                    features = []
                    for value in combined_df['soliciter_muac_cm']:
                        if pd.isna(value):
                            features.append([0])  # Handle missing values
                        else:
                            z_score = (value - real_mean) / (real_std + 1e-8)
                            features.append([z_score])
                    
                    return np.array(features), ['muac_z_score_real'], labels
        
        elif feature_type == 'muac_advanced':
            # Advanced MUAC features: z-scores from both populations + percentiles
            if 'soliciter_muac_cm' in combined_df.columns:
                real_muac = real_df['soliciter_muac_cm'].dropna()
                fake_muac = fake_df['soliciter_muac_cm'].dropna()
                
                if len(real_muac) > 0 and len(fake_muac) > 0:
                    real_mean, real_std = real_muac.mean(), real_muac.std()
                    fake_mean, fake_std = fake_muac.mean(), fake_muac.std()
                    real_sorted = np.sort(real_muac.values)
                    
                    features = []
                    for value in combined_df['soliciter_muac_cm']:
                        if pd.isna(value):
                            features.append([0, 0, 0])
                        else:
                            z_real = (value - real_mean) / (real_std + 1e-8)
                            z_fake = (value - fake_mean) / (fake_std + 1e-8)
                            percentile_real = np.searchsorted(real_sorted, value) / len(real_sorted)
                            features.append([z_real, z_fake, percentile_real])
                    
                    return np.array(features), ['muac_z_real', 'muac_z_fake', 'muac_percentile_real'], labels
        
        elif feature_type == 'age_basic':
            # Basic age z-score
            if 'childs_age_in_month' in combined_df.columns:
                real_age = real_df['childs_age_in_month'].dropna()
                if len(real_age) > 0:
                    real_mean = real_age.mean()
                    real_std = real_age.std()
                    
                    features = []
                    for value in combined_df['childs_age_in_month']:
                        if pd.isna(value):
                            features.append([0])
                        else:
                            z_score = (value - real_mean) / (real_std + 1e-8)
                            features.append([z_score])
                    
                    return np.array(features), ['age_z_score_real'], labels
        
        elif feature_type == 'age_advanced':
            # Advanced age features
            if 'childs_age_in_month' in combined_df.columns:
                real_age = real_df['childs_age_in_month'].dropna()
                fake_age = fake_df['childs_age_in_month'].dropna()
                
                if len(real_age) > 0 and len(fake_age) > 0:
                    real_mean, real_std = real_age.mean(), real_age.std()
                    fake_mean, fake_std = fake_age.mean(), fake_age.std()
                    
                    features = []
                    for value in combined_df['childs_age_in_month']:
                        if pd.isna(value):
                            features.append([0, 0])
                        else:
                            z_real = (value - real_mean) / (real_std + 1e-8)
                            z_fake = (value - fake_mean) / (fake_std + 1e-8)
                            features.append([z_real, z_fake])
                    
                    return np.array(features), ['age_z_real', 'age_z_fake'], labels
        
        elif feature_type == 'household_basic':
            # Basic household size feature
            if 'no_of_children' in combined_df.columns:
                real_household = real_df['no_of_children'].dropna()
                if len(real_household) > 0:
                    real_mean = real_household.mean()
                    real_std = real_household.std()
                    
                    features = []
                    for value in combined_df['no_of_children']:
                        if pd.isna(value):
                            features.append([0])
                        else:
                            z_score = (value - real_mean) / (real_std + 1e-8)
                            features.append([z_score])
                    
                    return np.array(features), ['household_z_score_real'], labels
        
        elif feature_type == 'muac_outlier':
            # MUAC outlier detection using IQR
            if 'soliciter_muac_cm' in combined_df.columns:
                real_muac = real_df['soliciter_muac_cm'].dropna()
                if len(real_muac) > 0:
                    q25, q75 = real_muac.quantile([0.25, 0.75])
                    iqr = q75 - q25
                    
                    features = []
                    for value in combined_df['soliciter_muac_cm']:
                        if pd.isna(value):
                            features.append([0])
                        else:
                            outlier_score = max(0, (value - q75) / (iqr + 1e-8)) if value > q75 else max(0, (q25 - value) / (iqr + 1e-8))
                            features.append([outlier_score])
                    
                    return np.array(features), ['muac_outlier_score'], labels
        
        elif feature_type == 'age_outlier':
            # Age outlier detection using IQR
            if 'childs_age_in_month' in combined_df.columns:
                real_age = real_df['childs_age_in_month'].dropna()
                if len(real_age) > 0:
                    q25, q75 = real_age.quantile([0.25, 0.75])
                    iqr = q75 - q25
                    
                    features = []
                    for value in combined_df['childs_age_in_month']:
                        if pd.isna(value):
                            features.append([0])
                        else:
                            outlier_score = max(0, (value - q75) / (iqr + 1e-8)) if value > q75 else max(0, (q25 - value) / (iqr + 1e-8))
                            features.append([outlier_score])
                    
                    return np.array(features), ['age_outlier_score'], labels
        
        elif feature_type == 'muac_percentile':
            # MUAC percentile position in real population
            if 'soliciter_muac_cm' in combined_df.columns:
                real_muac = real_df['soliciter_muac_cm'].dropna()
                if len(real_muac) > 0:
                    real_sorted = np.sort(real_muac.values)
                    
                    features = []
                    for value in combined_df['soliciter_muac_cm']:
                        if pd.isna(value):
                            features.append([0])
                        else:
                            percentile = np.searchsorted(real_sorted, value) / len(real_sorted)
                            features.append([percentile])
                    
                    return np.array(features), ['muac_percentile'], labels
        
        elif feature_type == 'age_percentile':
            # Age percentile position in real population
            if 'childs_age_in_month' in combined_df.columns:
                real_age = real_df['childs_age_in_month'].dropna()
                if len(real_age) > 0:
                    real_sorted = np.sort(real_age.values)
                    
                    features = []
                    for value in combined_df['childs_age_in_month']:
                        if pd.isna(value):
                            features.append([0])
                        else:
                            percentile = np.searchsorted(real_sorted, value) / len(real_sorted)
                            features.append([percentile])
                    
                    return np.array(features), ['age_percentile'], labels
        
        return None, [], labels
    
    def evaluate_feature_set(self, features, labels, feature_names, real_df=None, fake_df=None):
        """Evaluate a feature set with both visit-level and worker-level metrics."""
        if features is None or features.shape[1] == 0:
            return {'visit_auc': 0.0, 'user_auc': 0.0, 'visit_metrics': {}, 'user_metrics': {}}
        
        # Handle missing values
        features = np.nan_to_num(features, 0)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train classifier
        classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        
        # Train-test split for detailed evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        classifier.fit(X_train, y_train)
        
        # Visit-level predictions
        visit_probs = classifier.predict_proba(X_test)[:, 1]
        visit_auc = roc_auc_score(y_test, visit_probs)
        
        # Visit-level metrics at different thresholds
        visit_metrics = {}
        thresholds = [0.3, 0.5, 0.7]
        for threshold in thresholds:
            visit_preds = (visit_probs >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, visit_preds).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            visit_metrics[threshold] = {
                'sensitivity': sensitivity,
                'specificity': specificity
            }
        
        # Worker-level evaluation (if data provided)
        user_auc = 0.0
        user_metrics = {}
        
        if real_df is not None and fake_df is not None:
            # Get predictions for all data
            all_probs = classifier.predict_proba(features_scaled)[:, 1]
            
            # Create combined dataframe with predictions
            combined_df = pd.concat([real_df, fake_df], ignore_index=True)
            combined_df['fake_probability'] = all_probs
            combined_df['true_label'] = labels
            
            # Aggregate by worker
            user_stats = combined_df.groupby('flw_id').agg({
                'fake_probability': 'mean',
                'true_label': 'first'
            })
            
            user_probs = user_stats['fake_probability'].values
            user_labels = user_stats['true_label'].values
            
            if len(np.unique(user_labels)) > 1:  # Need both classes
                user_auc = roc_auc_score(user_labels, user_probs)
                
                # User-level metrics at different thresholds
                for threshold in thresholds:
                    user_preds = (user_probs >= threshold).astype(int)
                    tn, fp, fn, tp = confusion_matrix(user_labels, user_preds).ravel()
                    
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    user_metrics[threshold] = {
                        'sensitivity': sensitivity,
                        'specificity': specificity
                    }
        
        return {
            'visit_auc': visit_auc,
            'user_auc': user_auc,
            'visit_metrics': visit_metrics,
            'user_metrics': user_metrics
        }
    
    def incremental_feature_selection(self, real_df, fake_df):
        """
        Build features incrementally, showing value of each addition.
        """
        print("Comprehensive Feature Analysis")
        print("="*60)
        
        # Define ALL feature types to test
        feature_types = [
            ('muac_basic', 'MUAC Z-Score (vs Real Population)'),
            ('muac_advanced', 'MUAC Advanced (Z-Scores + Percentiles)'),
            ('age_basic', 'Child Age Z-Score'),
            ('age_advanced', 'Child Age Advanced (Dual Z-Scores)'),
            ('household_basic', 'Household Size Z-Score'),
            ('muac_outlier', 'MUAC Outlier Detection'),
            ('age_outlier', 'Age Outlier Detection'),
            ('muac_percentile', 'MUAC Percentile Position'),
            ('age_percentile', 'Age Percentile Position')
        ]
        
        # STEP 1: Test all features individually
        print("\nSTEP 1: INDIVIDUAL FEATURE PERFORMANCE")
        print("="*60)
        
        individual_results = []
        
        for feature_type, description in feature_types:
            print(f"\nTesting: {description}")
            
            features, names, labels = self.extract_single_feature_type(real_df, fake_df, feature_type)
            
            if features is None:
                print(f"  Skipped: No valid features extracted")
                continue
            
            # Evaluate individual performance
            performance = self.evaluate_feature_set(features, labels, names, real_df, fake_df)
            
            individual_results.append({
                'feature_type': feature_type,
                'description': description,
                'feature_names': names,
                'visit_auc': performance['visit_auc'],
                'user_auc': performance['user_auc'],
                'visit_metrics': performance['visit_metrics'],
                'user_metrics': performance['user_metrics'],
                'features': features
            })
            
            print(f"  Visit-level AUC: {performance['visit_auc']:.3f}")
            print(f"  Worker-level AUC: {performance['user_auc']:.3f}")
        
        # Sort by best overall performance (prioritize user-level, then visit-level)
        individual_results.sort(key=lambda x: (x['user_auc'], x['visit_auc']), reverse=True)
        
        print(f"\n" + "="*60)
        print("INDIVIDUAL FEATURE RANKING")
        print("="*60)
        
        for i, result in enumerate(individual_results, 1):
            print(f"{i}. {result['description']}")
            print(f"   Visit AUC: {result['visit_auc']:.3f} | Worker AUC: {result['user_auc']:.3f}")
        
        # STEP 2: Incremental building starting from best feature
        print(f"\n" + "="*60)
        print("STEP 2: INCREMENTAL FEATURE BUILDING")
        print("="*60)
        
        # Track cumulative features and performance
        cumulative_features = None
        cumulative_feature_names = []
        baseline_visit_auc = 0.0
        baseline_user_auc = 0.0
        
        incremental_results = []
        
        # Start with best performing feature
        for i, result in enumerate(individual_results):
            feature_type = result['feature_type']
            description = result['description']
            new_features = result['features']
            new_names = result['feature_names']
            
            print(f"\nTesting: {description}")
            
            # Combine with existing features
            if cumulative_features is None:
                combined_features = new_features
                combined_names = new_names
            else:
                combined_features = np.column_stack([cumulative_features, new_features])
                combined_names = cumulative_feature_names + new_names
            
            # Test combined performance
            combined_performance = self.evaluate_feature_set(combined_features, labels, combined_names, real_df, fake_df)
            
            visit_improvement = combined_performance['visit_auc'] - baseline_visit_auc
            user_improvement = combined_performance['user_auc'] - baseline_user_auc
            
            print(f"  Solo: Visit {result['visit_auc']:.3f} | Worker {result['user_auc']:.3f}")
            print(f"  Combined: Visit {combined_performance['visit_auc']:.3f} | Worker {combined_performance['user_auc']:.3f}")
            print(f"  Improvement: Visit +{visit_improvement:.3f} | Worker +{user_improvement:.3f}")
            
            # Decision: keep or discard (prioritize user improvement, but consider visit too)
            significant_improvement = (user_improvement >= self.min_improvement or 
                                     (user_improvement >= 0 and visit_improvement >= self.min_improvement))
            
            if significant_improvement or len(cumulative_feature_names) == 0:
                print(f"  ✅ KEPT: Significant improvement detected")
                cumulative_features = combined_features
                cumulative_feature_names = combined_names
                baseline_visit_auc = combined_performance['visit_auc']
                baseline_user_auc = combined_performance['user_auc']
                
                incremental_results.append({
                    'feature_type': feature_type,
                    'description': description,
                    'feature_names': new_names,
                    'solo_visit_auc': result['visit_auc'],
                    'solo_user_auc': result['user_auc'],
                    'combined_visit_auc': combined_performance['visit_auc'],
                    'combined_user_auc': combined_performance['user_auc'],
                    'visit_improvement': visit_improvement,
                    'user_improvement': user_improvement,
                    'visit_metrics': combined_performance['visit_metrics'],
                    'user_metrics': combined_performance['user_metrics'],
                    'kept': True,
                    'cumulative_features': len(combined_names)
                })
            else:
                print(f"  ❌ REJECTED: Insufficient improvement")
                incremental_results.append({
                    'feature_type': feature_type,
                    'description': description,
                    'feature_names': new_names,
                    'solo_visit_auc': result['visit_auc'],
                    'solo_user_auc': result['user_auc'],
                    'combined_visit_auc': baseline_visit_auc,
                    'combined_user_auc': baseline_user_auc,
                    'visit_improvement': visit_improvement,
                    'user_improvement': user_improvement,
                    'visit_metrics': {},
                    'user_metrics': {},
                    'kept': False,
                    'cumulative_features': len(cumulative_feature_names)
                })
        
        print(f"\n" + "="*60)
        print("FINAL FEATURE SET")
        print("="*60)
        print(f"Selected features ({len(cumulative_feature_names)}): {cumulative_feature_names}")
        print(f"Final performance: Visit {baseline_visit_auc:.3f} | Worker {baseline_user_auc:.3f} AUC-ROC")
        
        return individual_results, incremental_results, cumulative_features, cumulative_feature_names, labels
    
    def generate_comprehensive_report(self, individual_results, incremental_results):
        """Generate detailed report of incremental feature selection."""
        
        report = f"""
# Comprehensive Feature Analysis Report - Statistical Distribution Method

## Analysis Overview

This report analyzes **ALL potential features individually**, then builds the optimal feature set incrementally. We test both **individual visit detection** and **worker-level detection** performance.

**Strategy**: Start with best individual performer, add features only if they provide ≥{self.min_improvement:.1%} improvement.

## STEP 1: Individual Feature Performance

All features tested independently to identify the strongest signals:

| Rank | Feature | Visit AUC | Worker AUC | Primary Strength |
|------|---------|-----------|------------|------------------|"""

        for i, result in enumerate(individual_results, 1):
            primary = "Worker" if result['user_auc'] > result['visit_auc'] else "Visit"
            report += f"""
| {i} | {result['description']} | {result['visit_auc']:.3f} | {result['user_auc']:.3f} | {primary} |"""

        report += f"""

## STEP 2: Incremental Feature Building

Starting with the best performer, we add features that provide significant improvement:

| Step | Feature Added | Visit AUC | Worker AUC | Visit Improvement | Worker Improvement | Decision |
|------|---------------|-----------|------------|-------------------|-------------------|----------|"""

        for i, result in enumerate(incremental_results, 1):
            decision = "✅ KEPT" if result['kept'] else "❌ REJECTED"
            report += f"""
| {i} | {result['description']} | {result['combined_visit_auc']:.3f} | {result['combined_user_auc']:.3f} | +{result['visit_improvement']:.3f} | +{result['user_improvement']:.3f} | {decision} |"""

        # Get final performance from last kept feature
        kept_features = [r for r in incremental_results if r['kept']]
        final_visit_auc = kept_features[-1]['combined_visit_auc'] if kept_features else 0
        final_user_auc = kept_features[-1]['combined_user_auc'] if kept_features else 0
        
        # Performance tables for final feature set
        if kept_features:
            final_result = kept_features[-1]
            
            report += f"""

## Final Performance - Individual Visit Detection

**Dataset**: 6,000 real visits + 2,003 fake visits  
**Overall Performance**: {final_visit_auc:.1%} accuracy (AUC-ROC = {final_visit_auc:.3f})

| Threshold | Fake Visits Caught | Real Visits Correctly Identified | False Alarms | Sensitivity | Specificity |
|-----------|-------------------|----------------------------------|---------------|-------------|-------------|"""

            for threshold in [0.3, 0.5, 0.7]:
                if threshold in final_result['visit_metrics']:
                    metrics = final_result['visit_metrics'][threshold]
                    sens = metrics['sensitivity']
                    spec = metrics['specificity']
                    false_alarms = 100 - spec * 100
                    
                    report += f"""
| **{threshold}** | **{sens:.1%}** | {spec:.1%} | {false_alarms:.1f}% | {sens:.3f} | {spec:.3f} |"""

            report += f"""

## Final Performance - Worker-Level Detection

**Dataset**: 380 real workers + 18 fake workers  
**Overall Performance**: {final_user_auc:.1%} accuracy (AUC-ROC = {final_user_auc:.3f})

| Threshold | Fake Workers Caught | Real Workers Correctly Identified | False Accusations | Sensitivity | Specificity |
|-----------|-------------------|----------------------------------|------------------|-------------|-------------|"""

            for threshold in [0.3, 0.5, 0.7]:
                if threshold in final_result['user_metrics']:
                    metrics = final_result['user_metrics'][threshold]
                    sens = metrics['sensitivity']
                    spec = metrics['specificity']
                    false_accusations = 100 - spec * 100
                    
                    report += f"""
| **{threshold}** | **{sens:.1%}** | {spec:.1%} | {false_accusations:.1f}% | {sens:.3f} | {spec:.3f} |"""

        report += f"""

## Key Insights

**Champion Feature**: {individual_results[0]['description']} - {individual_results[0]['user_auc']:.1%} worker detection alone

**Feature Efficiency**: {len(kept_features)} features achieve {final_user_auc:.1%} worker detection accuracy

**Optimal Use Cases**:
- **Worker Monitoring**: Use for identifying problematic workers (excellent {final_user_auc:.1%} accuracy)
- **Visit Screening**: Use for real-time quality control ({final_visit_auc:.1%} accuracy)

## Selected Feature Set

**Final Features**: {len(kept_features)} types, {kept_features[-1]['cumulative_features'] if kept_features else 0} total features

"""
        
        for i, result in enumerate(kept_features, 1):
            report += f"""
**{i}. {result['description']}**
- Solo: Visit {result['solo_visit_auc']:.3f} | Worker {result['solo_user_auc']:.3f}
- Improvement: Visit +{result['visit_improvement']:.3f} | Worker +{result['user_improvement']:.3f}
- Features: {', '.join(result['feature_names'])}
"""

        report += f"""
## Implementation Notes

- **Simplicity**: Only {len(kept_features)} feature calculations needed
- **Speed**: Extremely fast - basic statistical calculations only  
- **Interpretability**: Clear mathematical basis for each decision
- **Robustness**: Focuses on content quality, not volume artifacts

"""
        
        return report

def main():
    """Main function to run incremental feature selection."""
    
    # Load data
    real_df, fake_df, combined_df = get_analysis_datasets()
    
    # Sample for development speed
    sample_size = min(len(fake_df) * 3, 6000)
    real_sample = real_df.sample(n=sample_size, random_state=42)
    
    print(f"Dataset: {len(real_sample):,} real + {len(fake_df):,} fake visits")
    
    # Run comprehensive feature analysis
    detector = IncrementalFeatureDetector(min_improvement=0.01)  # 1% minimum improvement
    individual_results, incremental_results, final_features, final_names, labels = detector.incremental_feature_selection(
        real_sample, fake_df
    )
    
    # Generate comprehensive report
    report = detector.generate_comprehensive_report(individual_results, incremental_results)
    
    # Save report with timestamp
    import os
    from datetime import datetime
    
    os.makedirs('output', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'output/comprehensive_feature_analysis_report_{timestamp}.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n" + "="*50)
    print("INCREMENTAL FEATURE REPORT GENERATED")
    print("="*50)
    print(f"Report saved to: {report_path}")
    
    return individual_results, incremental_results, final_features, final_names

if __name__ == "__main__":
    main()
