#!/usr/bin/env python3
"""
Proper classification approach:
- Single features: Simple thresholds only
- Multiple features: ML models for complex interactions
- Clean, minimal output
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def analyze_single_features(features, labels, feature_names):
    """
    Analyze single features using simple thresholds only.
    This is the right approach for single-feature analysis.
    """
    print(f"\n🎯 SINGLE FEATURE ANALYSIS (Threshold-Based)")
    print("="*60)
    
    results = []
    
    for i, feature_name in enumerate(feature_names):
        feature_values = features[:, i]
        
        # Skip features with no variation
        if len(np.unique(feature_values)) < 2:
            continue
            
        # Find optimal threshold
        best_threshold, best_auc, best_direction, best_predictions = find_optimal_threshold(feature_values, labels)
        
        # Calculate actual detection performance
        fake_detected = np.sum(best_predictions[labels == 1])  # True positives
        total_fake = np.sum(labels == 1)
        real_flagged = np.sum(best_predictions[labels == 0])   # False positives
        total_real = np.sum(labels == 0)
        
        fake_detection_rate = fake_detected / total_fake if total_fake > 0 else 0
        false_positive_rate = real_flagged / total_real if total_real > 0 else 0
        
        # Store results
        results.append({
            'feature': feature_name,
            'auc': best_auc,
            'threshold': best_threshold,
            'direction': best_direction,
            'real_mean': np.mean(feature_values[labels == 0]),
            'fake_mean': np.mean(feature_values[labels == 1]),
            'fake_detection_rate': fake_detection_rate,
            'false_positive_rate': false_positive_rate,
            'fake_detected': fake_detected,
            'total_fake': total_fake,
            'real_flagged': real_flagged,
            'total_real': total_real
        })
    
    # Sort by AUC
    results.sort(key=lambda x: x['auc'], reverse=True)
    
    # Print top 10 results with actual detection performance
    print(f"\n📊 TOP 10 SINGLE FEATURES (Actual Detection Performance):")
    print(f"{'Rank':<4} {'Feature':<30} {'AUC':<6} {'Fake Detected':<13} {'False Pos':<10} {'Rule':<25}")
    print("-" * 95)
    
    for i, result in enumerate(results[:10]):
        fake_pct = result['fake_detection_rate'] * 100
        fp_pct = result['false_positive_rate'] * 100
        rule = f"{result['direction']} {result['threshold']:.2f}"
        print(f"{i+1:<4} {result['feature'][:29]:<30} {result['auc']:.3f}  {fake_pct:>5.1f}% ({result['fake_detected']}/{result['total_fake']})  {fp_pct:>5.1f}% ({result['real_flagged']}/{result['total_real']})  {rule:<25}")
    
    # Show detailed performance for top feature
    if results:
        top = results[0]
        print(f"\n🎯 BEST FEATURE DETAILED PERFORMANCE:")
        print(f"   📊 Feature: {top['feature']}")
        print(f"   📈 AUC Score: {top['auc']:.3f}")
        print(f"   🎯 Rule: {top['feature']} {top['direction']} {top['threshold']:.3f}")
        print(f"   ✅ Fake Users Detected: {top['fake_detected']}/{top['total_fake']} ({top['fake_detection_rate']*100:.1f}%)")
        print(f"   ⚠️  Real Users Flagged: {top['real_flagged']}/{top['total_real']} ({top['false_positive_rate']*100:.1f}%)")
        
        if top['fake_detection_rate'] >= 0.8:
            print(f"   🏆 EXCELLENT: Catches {top['fake_detection_rate']*100:.0f}% of fake users!")
        elif top['fake_detection_rate'] >= 0.6:
            print(f"   👍 GOOD: Catches {top['fake_detection_rate']*100:.0f}% of fake users")
        else:
            print(f"   ⚠️  MODERATE: Only catches {top['fake_detection_rate']*100:.0f}% of fake users")
    
    return results

def analyze_multi_features(features, labels, feature_names, single_results):
    """
    Use ML models for multi-feature analysis.
    This is where ML actually adds value.
    """
    print(f"\n🤖 MULTI-FEATURE ANALYSIS (ML Models)")
    print("="*60)
    
    # Select top features (AUC > 0.7)
    top_features = [r for r in single_results if r['auc'] > 0.7]
    
    if len(top_features) < 2:
        print("⚠️  Not enough strong features (AUC > 0.7) for multi-feature analysis")
        return {}
    
    # Get feature indices and data
    feature_indices = [feature_names.index(f['feature']) for f in top_features]
    X_multi = features[:, feature_indices]
    top_feature_names = [f['feature'] for f in top_features]
    
    print(f"🎯 Using {len(top_features)} top features:")
    for i, feat in enumerate(top_feature_names):
        print(f"   {i+1}. {feat} (AUC: {top_features[i]['auc']:.3f})")
    
    # Define ML models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    }
    
    results = {}
    
    print(f"\n🔄 Cross-Validation Results (5-fold):")
    print(f"{'Model':<20} {'Mean AUC':<10} {'Std AUC':<10}")
    print("-" * 40)
    
    for name, model in models.items():
        # 5-fold cross-validation
        cv_scores = cross_val_score(model, X_multi, labels, cv=5, scoring='roc_auc')
        mean_auc = cv_scores.mean()
        std_auc = cv_scores.std()
        
        results[name] = {
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'cv_scores': cv_scores
        }
        
        print(f"{name:<20} {mean_auc:.3f}     {std_auc:.3f}")
        
        # Get feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            model.fit(X_multi, labels)
            importances = model.feature_importances_
            results[name]['feature_importance'] = list(zip(top_feature_names, importances))
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['mean_auc'])
    best_auc = results[best_model_name]['mean_auc']
    best_single_auc = max(r['auc'] for r in single_results)
    
    print(f"\n🏆 BEST MULTI-FEATURE MODEL: {best_model_name}")
    print(f"   📊 AUC: {best_auc:.3f} ± {results[best_model_name]['std_auc']:.3f}")
    print(f"   🎯 Improvement over best single feature: +{(best_auc - best_single_auc):.3f}")
    
    # Show feature importance for best model
    if 'feature_importance' in results[best_model_name]:
        print(f"   🔝 Feature Importance:")
        for feat, imp in sorted(results[best_model_name]['feature_importance'], key=lambda x: x[1], reverse=True):
            print(f"      - {feat}: {imp:.3f}")
    
    return results

def find_optimal_threshold(feature_values, labels):
    """Find the threshold that maximizes AUC for a single feature."""
    # Get unique values as threshold candidates
    thresholds = np.unique(feature_values)
    best_auc = 0
    best_threshold = thresholds[0]
    best_direction = '>='
    best_predictions = None
    
    for threshold in thresholds:
        # Try both directions (>= and <=)
        for direction in ['>=', '<=']:
            if direction == '>=':
                predictions = (feature_values >= threshold).astype(int)
            else:
                predictions = (feature_values <= threshold).astype(int)
            
            # Calculate AUC
            try:
                auc = roc_auc_score(labels, predictions)
                if auc > best_auc:
                    best_auc = auc
                    best_threshold = threshold
                    best_direction = direction
                    best_predictions = predictions
            except:
                continue
    
    return best_threshold, best_auc, best_direction, best_predictions

def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate comprehensive classification metrics.
    Consolidated from classification_runner.py for completeness.
    """
    metrics = {}
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Basic metrics
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = metrics['sensitivity']
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    # AUC (if probabilities available)
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    else:
        metrics['auc'] = 0.5  # Random performance
    
    # Confusion matrix components
    metrics['true_positives'] = tp
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    
    # Threshold-specific metrics
    metrics['thresholds'] = {}
    if y_proba is not None:
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            thresh_pred = (y_proba >= threshold).astype(int)
            try:
                tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, thresh_pred).ravel()
                
                metrics['thresholds'][threshold] = {
                    'sensitivity': tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0,
                    'specificity': tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0,
                    'precision': tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0,
                    'false_positive_rate': fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
                }
            except:
                # Handle edge cases where threshold creates empty classes
                continue
    
    return metrics

def perform_generic_classification(features, labels, feature_names, classifiers, analysis_type="classification"):
    """
    Generic ML classification utility.
    Consolidated from classification_runner.py for reusability.
    """
    print(f"⚡ Running {analysis_type}-level classification...")
    
    # Handle edge cases
    if features.shape[0] == 0 or features.shape[1] == 0:
        print("❌ No features available for classification")
        return {}
    
    if len(np.unique(labels)) < 2:
        print("❌ Need both fake and real samples for classification")
        return {}
    
    # Prepare features
    features_clean = np.nan_to_num(features, 0)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, 
        test_size=0.3, 
        random_state=42, 
        stratify=labels
    )
    
    print(f"📊 Dataset split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    print(f"   Features: {features.shape[1]}")
    
    results = {}
    
    # Test each classifier
    for clf_name, classifier in classifiers.items():
        print(f"\n🔍 Testing {clf_name}...")
        
        try:
            # Train classifier
            classifier.fit(X_train, y_train)
            
            # Predictions
            y_pred = classifier.predict(X_test)
            y_proba = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, 'predict_proba') else None
            
            # Calculate metrics
            clf_results = calculate_comprehensive_metrics(y_test, y_pred, y_proba)
            
            # Feature importance (if available)
            if hasattr(classifier, 'feature_importances_'):
                clf_results['feature_importance'] = dict(zip(feature_names, classifier.feature_importances_))
            elif hasattr(classifier, 'coef_'):
                # For linear models, use absolute coefficients
                clf_results['feature_importance'] = dict(zip(feature_names, np.abs(classifier.coef_[0])))
            else:
                clf_results['feature_importance'] = {}
            
            # Cross-validation score
            cv_scores = cross_val_score(classifier, features_scaled, labels, cv=5, scoring='roc_auc')
            clf_results['cv_auc_mean'] = cv_scores.mean()
            clf_results['cv_auc_std'] = cv_scores.std()
            
            results[clf_name] = clf_results
            
            print(f"   AUC: {clf_results['auc']:.3f}")
            print(f"   CV AUC: {clf_results['cv_auc_mean']:.3f} ± {clf_results['cv_auc_std']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            results[clf_name] = {'error': str(e)}
    
    # Find best classifier
    valid_results = {name: res for name, res in results.items() if 'auc' in res}
    if valid_results:
        best_clf = max(valid_results.keys(), key=lambda x: valid_results[x]['auc'])
        print(f"\n🏆 Best {analysis_type}-level classifier: {best_clf} (AUC: {valid_results[best_clf]['auc']:.3f})")
        results['best_classifier'] = best_clf
    
    return results

def run_classification_analysis(features, labels, feature_names, analysis_type="worker"):
    """
    Main function to run proper classification analysis.
    """
    print(f"\n🎯 {analysis_type.upper()}-LEVEL CLASSIFICATION ANALYSIS")
    print("="*80)
    
    # Step 1: Single feature analysis (thresholds only)
    single_results = analyze_single_features(features, labels, feature_names)
    
    # Step 2: Multi-feature analysis (ML models)
    multi_results = analyze_multi_features(features, labels, feature_names, single_results)
    
    # Step 3: Summary and recommendations
    print(f"\n📋 SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    best_single = single_results[0]
    print(f"🥇 Best Single Feature: {best_single['feature']}")
    print(f"   📊 AUC: {best_single['auc']:.3f}")
    print(f"   🎯 Rule: {best_single['feature']} {best_single['direction']} {best_single['threshold']:.3f}")
    print(f"   ✅ Detects: {best_single['fake_detected']}/{best_single['total_fake']} fake users ({best_single['fake_detection_rate']*100:.1f}%)")
    print(f"   ⚠️  Flags: {best_single['real_flagged']}/{best_single['total_real']} real users ({best_single['false_positive_rate']*100:.1f}%)")
    
    if multi_results:
        best_multi_name = max(multi_results.keys(), key=lambda x: multi_results[x]['mean_auc'])
        best_multi_auc = multi_results[best_multi_name]['mean_auc']
        
        print(f"\n🤖 Best Multi-Feature Model: {best_multi_name}")
        print(f"   📊 AUC: {best_multi_auc:.3f}")
        
        if best_multi_auc > best_single['auc'] + 0.05:  # 5% improvement threshold
            print(f"   ✅ RECOMMENDATION: Use multi-feature model (+{(best_multi_auc - best_single['auc']):.3f} AUC improvement)")
        else:
            print(f"   ✅ RECOMMENDATION: Use simple single-feature rule (easier to interpret, similar performance)")
    
    return {
        'single_results': single_results,
        'multi_results': multi_results,
        'analysis_type': analysis_type
    }

if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Create dummy features
    features = np.random.randn(n_samples, n_features)
    labels = np.random.choice([0, 1], n_samples)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Run analysis
    results = run_classification_analysis(features, labels, feature_names)
