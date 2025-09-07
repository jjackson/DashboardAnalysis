#!/usr/bin/env python3
"""
Detailed model evaluation - shows precision, recall, and false positive rates
for multi-feature models at different thresholds.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from data_loader import load_data
from feature_extractor import extract_worker_features

def evaluate_model_thresholds(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance at different probability thresholds.
    """
    # Get prediction probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n📊 {model_name} Performance at Different Thresholds:")
    print("Threshold | Precision | Recall | False Pos Rate | Fake Detected | Real Flagged")
    print("-" * 80)
    
    results = []
    
    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((y_test == 1) & (y_pred == 1))  # True positives
        fp = np.sum((y_test == 0) & (y_pred == 1))  # False positives
        tn = np.sum((y_test == 0) & (y_pred == 0))  # True negatives
        fn = np.sum((y_test == 1) & (y_pred == 0))  # False negatives
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_pos_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Count totals
        fake_total = np.sum(y_test == 1)
        real_total = np.sum(y_test == 0)
        fake_detected = tp
        real_flagged = fp
        
        print(f"{threshold:8.1f} | {precision:8.1%} | {recall:7.1%} | {false_pos_rate:13.1%} | "
              f"{fake_detected:4d}/{fake_total:2d} ({recall:5.1%}) | {real_flagged:4d}/{real_total:3d} ({false_pos_rate:5.1%})")
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'false_pos_rate': false_pos_rate,
            'fake_detected': fake_detected,
            'fake_total': fake_total,
            'real_flagged': real_flagged,
            'real_total': real_total
        })
    
    return results

def find_best_threshold(results, max_false_pos_rate=0.10):
    """
    Find the best threshold that keeps false positive rate below threshold.
    """
    valid_thresholds = [r for r in results if r['false_pos_rate'] <= max_false_pos_rate]
    
    if not valid_thresholds:
        print(f"\n⚠️  No threshold achieves false positive rate <= {max_false_pos_rate:.1%}")
        return None
    
    # Among valid thresholds, pick the one with highest recall
    best = max(valid_thresholds, key=lambda x: x['recall'])
    
    print(f"\n🎯 RECOMMENDED THRESHOLD (FPR <= {max_false_pos_rate:.1%}):")
    print(f"   Threshold: {best['threshold']:.1f}")
    print(f"   Precision: {best['precision']:.1%}")
    print(f"   Recall: {best['recall']:.1%}")
    print(f"   False Positive Rate: {best['false_pos_rate']:.1%}")
    print(f"   Fake Workers Detected: {best['fake_detected']}/{best['fake_total']} ({best['recall']:.1%})")
    print(f"   Real Workers Flagged: {best['real_flagged']}/{best['real_total']} ({best['false_pos_rate']:.1%})")
    
    return best

def main():
    """
    Run detailed evaluation of multi-feature models.
    """
    print("🎯 DETAILED MULTI-FEATURE MODEL EVALUATION")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading data...")
    real_df, fake_df = load_data()
    
    # Extract worker features
    print("\n🔍 Extracting worker features...")
    worker_features, worker_labels, worker_feature_names = extract_worker_features(real_df, fake_df)
    
    print(f"Dataset: {len(worker_labels)} workers ({np.sum(worker_labels)} fake, {np.sum(worker_labels == 0)} real)")
    
    # Prepare data
    X = np.nan_to_num(worker_features, 0)
    y = worker_labels
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} workers")
    print(f"Test set: {len(X_test)} workers ({np.sum(y_test)} fake, {np.sum(y_test == 0)} real)")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Train and evaluate each model
    best_models = {}
    
    for model_name, model in models.items():
        print(f"\n" + "=" * 60)
        print(f"🤖 EVALUATING {model_name.upper()}")
        print("=" * 60)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Calculate AUC
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"AUC Score: {auc:.3f}")
        
        # Evaluate at different thresholds
        results = evaluate_model_thresholds(model, X_test, y_test, model_name)
        
        # Find best threshold for 10% false positive rate
        best_threshold = find_best_threshold(results, max_false_pos_rate=0.10)
        
        if best_threshold:
            best_models[model_name] = {
                'model': model,
                'auc': auc,
                'best_threshold': best_threshold,
                'scaler': scaler,
                'feature_names': worker_feature_names
            }
    
    # Compare models
    if best_models:
        print(f"\n" + "=" * 60)
        print("🏆 MODEL COMPARISON (FPR <= 10%)")
        print("=" * 60)
        
        print("Model               | AUC   | Threshold | Recall | Precision | FPR")
        print("-" * 70)
        
        for name, info in best_models.items():
            bt = info['best_threshold']
            print(f"{name:<18} | {info['auc']:.3f} | {bt['threshold']:8.1f} | {bt['recall']:5.1%} | "
                  f"{bt['precision']:8.1%} | {bt['false_pos_rate']:4.1%}")
        
        # Recommend best model
        best_model_name = max(best_models.keys(), key=lambda x: best_models[x]['best_threshold']['recall'])
        best_info = best_models[best_model_name]
        
        print(f"\n🎯 PRODUCTION RECOMMENDATION: {best_model_name}")
        bt = best_info['best_threshold']
        print(f"   Use threshold: {bt['threshold']:.1f}")
        print(f"   Expected performance:")
        print(f"   - Detect {bt['recall']:.1%} of fake workers")
        print(f"   - Flag {bt['false_pos_rate']:.1%} of real workers as suspicious")
        print(f"   - {bt['precision']:.1%} of flagged workers will be truly fake")
        
        # Show feature importance
        if hasattr(best_info['model'], 'feature_importances_'):
            importances = best_info['model'].feature_importances_
            feature_importance = list(zip(best_info['feature_names'], importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
            for i, (feat, imp) in enumerate(feature_importance[:10]):
                print(f"   {i+1:2d}. {feat:<35} {imp:.3f}")

if __name__ == "__main__":
    main()

