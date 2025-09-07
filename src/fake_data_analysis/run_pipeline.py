#!/usr/bin/env python3
"""
Main analysis runner - clean and simple orchestration of fake data detection.
"""

from data_loader import load_data
from feature_extractor import extract_visit_features, extract_worker_features
from classification import run_classification_analysis

def main():
    """
    Main analysis pipeline - clean and modular.
    """
    print("🎯 FAKE DATA DETECTION ANALYSIS")
    print("=" * 50)
    
    # 1. Load and prepare data
    print("\n📊 STEP 1: Loading Data")
    real_df, fake_df = load_data()
    
    # 2. Extract visit-level features (one row per visit)
    print("\n🔍 STEP 2: Extracting Visit Features")
    visit_features, visit_labels, visit_feature_names = extract_visit_features(real_df, fake_df)
    
    # 3. Extract worker-level features (one row per worker)
    print("\n👥 STEP 3: Extracting Worker Features")
    worker_features, worker_labels, worker_feature_names = extract_worker_features(real_df, fake_df)
    
    # 4. Run Proper Classification Analysis
    print("\n⚡ STEP 4: Running Classification Analysis")
    
    # Analyze visit-level features (optional - usually worker-level is more important)
    print("\n" + "="*80)
    visit_results = run_classification_analysis(
        visit_features, visit_labels, visit_feature_names, 
        analysis_type="visit"
    )
    
    # Analyze worker-level features (main focus)
    print("\n" + "="*80)
    worker_results = run_classification_analysis(
        worker_features, worker_labels, worker_feature_names,
        analysis_type="worker"
    )
    
    # 5. Final Summary
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("="*80)
    
    # Show the best classifier for production use
    best_worker_feature = worker_results['single_results'][0]
    print(f"\n🎯 PRODUCTION CLASSIFIER RECOMMENDATION:")
    print(f"   📊 Feature: {best_worker_feature['feature']}")
    print(f"   📈 AUC: {best_worker_feature['auc']:.3f}")
    print(f"   🎯 Rule: Flag as suspicious if {best_worker_feature['feature']} {best_worker_feature['direction']} {best_worker_feature['threshold']:.3f}")
    
    print(f"\n📊 ACTUAL DETECTION PERFORMANCE:")
    print(f"   ✅ Fake Users Detected: {best_worker_feature['fake_detected']}/{best_worker_feature['total_fake']} ({best_worker_feature['fake_detection_rate']*100:.1f}%)")
    print(f"   ⚠️  Real Users Flagged: {best_worker_feature['real_flagged']}/{best_worker_feature['total_real']} ({best_worker_feature['false_positive_rate']*100:.1f}%)")
    
    # Generate the sentence you wanted
    final_sentence = (f"With a relatively simple classifier, we were able to identify "
                     f"{best_worker_feature['fake_detection_rate']*100:.0f}% of fake users. When applied to the entire real data set, "
                     f"this also found {best_worker_feature['false_positive_rate']*100:.1f}% of users which we are unsure if they "
                     f"really represent fake data or not.")
    
    print(f"\n💬 FINAL SUMMARY:")
    print(f'"{final_sentence}"')
    
    return visit_results, worker_results

if __name__ == "__main__":
    main()

