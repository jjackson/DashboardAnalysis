#!/usr/bin/env python3
"""
Report generator module - creates comprehensive analysis reports.
"""

import os
from datetime import datetime

def generate_analysis_report(visit_results, worker_results, visit_features, worker_features, 
                           real_count, fake_count):
    """
    Generate a comprehensive analysis report.
    
    Args:
        visit_results: Results from visit-level classification
        worker_results: Results from worker-level classification
        visit_features: List of visit-level feature names
        worker_features: List of worker-level feature names
        real_count: Number of real visits
        fake_count: Number of fake visits
        
    Returns:
        str: Path to generated report
    """
    print("📋 Generating comprehensive analysis report...")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'output/fake_data_analysis_report_{timestamp}.md'
    
    # Generate report content
    report = _create_report_content(
        visit_results, worker_results, visit_features, worker_features,
        real_count, fake_count, timestamp
    )
    
    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Report generated: {report_path}")
    return report_path

def _create_report_content(visit_results, worker_results, visit_features, worker_features,
                          real_count, fake_count, timestamp):
    """Create the main report content."""
    
    report = f"""# Fake Data Detection Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Analysis ID**: {timestamp}

## Executive Summary

This report presents a comprehensive analysis of fake data detection using machine learning classification on CommCare Connect data. The analysis separates visit-level detection (individual form submissions) from worker-level detection (identifying problematic workers).

### Dataset Overview
- **Real Data**: {real_count:,} visits from legitimate field workers
- **Fake Data**: {fake_count:,} visits from workers paid to create realistic fake data
- **Analysis Approach**: Proper separation of visit-level vs worker-level features and evaluation

---

## Visit-Level Analysis

**Objective**: Detect individual fake form submissions in real-time

### Dataset
- **Total Visits**: {real_count + fake_count:,} individual form submissions
- **Features**: {len(visit_features)} visit-level indicators

### Feature Set
"""
    
    # Visit-level features
    for i, feature in enumerate(visit_features, 1):
        report += f"{i}. `{feature}`\n"
    
    report += "\n### Classification Results\n\n"
    
    # Visit-level results table
    if visit_results:
        report += "| Classifier | AUC-ROC | CV AUC | Accuracy | Sensitivity | Specificity |\n"
        report += "|------------|---------|--------|----------|-------------|-------------|\n"
        
        for clf_name, results in visit_results.items():
            if clf_name == 'best_classifier' or 'error' in results:
                continue
            
            auc = results.get('auc', 0)
            cv_auc = results.get('cv_auc_mean', 0)
            accuracy = results.get('accuracy', 0)
            sensitivity = results.get('sensitivity', 0)
            specificity = results.get('specificity', 0)
            
            report += f"| {clf_name} | {auc:.3f} | {cv_auc:.3f} | {accuracy:.3f} | {sensitivity:.3f} | {specificity:.3f} |\n"
        
        # Best classifier details
        if 'best_classifier' in visit_results:
            best_clf = visit_results['best_classifier']
            best_results = visit_results[best_clf]
            
            report += f"\n### Best Visit-Level Classifier: {best_clf}\n\n"
            report += f"**Performance**: AUC-ROC = {best_results['auc']:.3f}\n\n"
            
            # Threshold analysis
            if 'thresholds' in best_results:
                report += "#### Performance at Different Thresholds\n\n"
                report += "| Threshold | Fake Visits Caught | Real Visits Correctly ID'd | False Alarms |\n"
                report += "|-----------|--------------------|-----------------------------|---------------|\n"
                
                for threshold in [0.3, 0.5, 0.7]:
                    if threshold in best_results['thresholds']:
                        thresh_metrics = best_results['thresholds'][threshold]
                        sensitivity = thresh_metrics['sensitivity']
                        specificity = thresh_metrics['specificity']
                        false_alarm_rate = 1 - specificity
                        
                        report += f"| {threshold} | {sensitivity:.1%} | {specificity:.1%} | {false_alarm_rate:.1%} |\n"
            
            # Feature importance
            if 'feature_importance' in best_results and best_results['feature_importance']:
                report += "\n#### Most Important Visit-Level Features\n\n"
                sorted_features = sorted(best_results['feature_importance'].items(), 
                                       key=lambda x: x[1], reverse=True)
                
                for i, (feature, importance) in enumerate(sorted_features[:5], 1):
                    report += f"{i}. **{feature}**: {importance:.3f}\n"
    
    report += "\n---\n\n## Worker-Level Analysis\n\n"
    report += "**Objective**: Identify workers with systematic fake data submission patterns\n\n"
    
    # Worker-level analysis
    worker_count = len(set(visit_results.get('worker_ids', []))) if 'worker_ids' in visit_results else "Unknown"
    report += f"### Dataset\n"
    report += f"- **Total Workers**: {worker_count} individual field workers\n"
    report += f"- **Features**: {len(worker_features)} worker-level behavioral indicators\n\n"
    
    # Worker-level features
    report += "### Feature Set\n\n"
    for i, feature in enumerate(worker_features, 1):
        report += f"{i}. `{feature}`\n"
    
    report += "\n### Classification Results\n\n"
    
    # Worker-level results table
    if worker_results:
        report += "| Classifier | AUC-ROC | CV AUC | Accuracy | Sensitivity | Specificity |\n"
        report += "|------------|---------|--------|----------|-------------|-------------|\n"
        
        for clf_name, results in worker_results.items():
            if clf_name == 'best_classifier' or 'error' in results:
                continue
            
            auc = results.get('auc', 0)
            cv_auc = results.get('cv_auc_mean', 0)
            accuracy = results.get('accuracy', 0)
            sensitivity = results.get('sensitivity', 0)
            specificity = results.get('specificity', 0)
            
            report += f"| {clf_name} | {auc:.3f} | {cv_auc:.3f} | {accuracy:.3f} | {sensitivity:.3f} | {specificity:.3f} |\n"
        
        # Best worker classifier
        if 'best_classifier' in worker_results:
            best_clf = worker_results['best_classifier']
            best_results = worker_results[best_clf]
            
            report += f"\n### Best Worker-Level Classifier: {best_clf}\n\n"
            report += f"**Performance**: AUC-ROC = {best_results['auc']:.3f}\n\n"
            
            # Feature importance
            if 'feature_importance' in best_results and best_results['feature_importance']:
                report += "#### Most Important Worker-Level Features\n\n"
                sorted_features = sorted(best_results['feature_importance'].items(), 
                                       key=lambda x: x[1], reverse=True)
                
                for i, (feature, importance) in enumerate(sorted_features[:5], 1):
                    report += f"{i}. **{feature}**: {importance:.3f}\n"
    
    # Key insights and recommendations
    report += "\n---\n\n## Key Insights\n\n"
    
    # Determine best approach
    visit_auc = 0
    worker_auc = 0
    
    if visit_results and 'best_classifier' in visit_results:
        visit_auc = visit_results[visit_results['best_classifier']].get('auc', 0)
    
    if worker_results and 'best_classifier' in worker_results:
        worker_auc = worker_results[worker_results['best_classifier']].get('auc', 0)
    
    if visit_auc > worker_auc:
        report += f"### 🎯 Visit-Level Detection is More Effective\n\n"
        report += f"- **Visit-level AUC**: {visit_auc:.3f} vs **Worker-level AUC**: {worker_auc:.3f}\n"
        report += f"- **Recommendation**: Focus on real-time visit screening\n"
        report += f"- **Use Case**: Flag suspicious individual form submissions as they arrive\n\n"
    else:
        report += f"### 👥 Worker-Level Detection is More Effective\n\n"
        report += f"- **Worker-level AUC**: {worker_auc:.3f} vs **Visit-level AUC**: {visit_auc:.3f}\n"
        report += f"- **Recommendation**: Focus on worker behavior monitoring\n"
        report += f"- **Use Case**: Identify problematic workers for investigation\n\n"
    
    # Top features across both levels
    all_features = {}
    
    if visit_results and 'best_classifier' in visit_results:
        best_visit = visit_results[visit_results['best_classifier']]
        if 'feature_importance' in best_visit:
            for feature, importance in best_visit['feature_importance'].items():
                all_features[f"Visit: {feature}"] = importance
    
    if worker_results and 'best_classifier' in worker_results:
        best_worker = worker_results[worker_results['best_classifier']]
        if 'feature_importance' in best_worker:
            for feature, importance in best_worker['feature_importance'].items():
                all_features[f"Worker: {feature}"] = importance
    
    if all_features:
        report += "### 🔍 Most Discriminative Features Overall\n\n"
        sorted_all = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_all[:8], 1):
            report += f"{i}. **{feature}**: {importance:.3f}\n"
    
    # Implementation recommendations
    report += "\n## Implementation Recommendations\n\n"
    
    if visit_auc > 0.8:
        report += "### ✅ Ready for Production - Visit-Level Detection\n\n"
        report += f"With {visit_auc:.1%} accuracy, visit-level detection is ready for deployment:\n\n"
        report += "1. **Real-time Screening**: Flag suspicious visits as they arrive\n"
        report += "2. **Quality Control**: Review flagged visits before data entry\n"
        report += "3. **Worker Feedback**: Provide immediate feedback on data quality\n\n"
    elif worker_auc > 0.8:
        report += "### ✅ Ready for Production - Worker-Level Detection\n\n"
        report += f"With {worker_auc:.1%} accuracy, worker-level detection is ready for deployment:\n\n"
        report += "1. **Worker Monitoring**: Regular assessment of worker data quality\n"
        report += "2. **Training Identification**: Target workers needing additional training\n"
        report += "3. **Investigation Triggers**: Flag workers for detailed review\n\n"
    else:
        report += "### ⚠️ Needs Improvement\n\n"
        report += "Current performance may not be sufficient for production deployment:\n\n"
        report += "1. **Feature Engineering**: Develop additional discriminative features\n"
        report += "2. **Data Collection**: Gather more training data\n"
        report += "3. **Domain Expertise**: Incorporate more medical/field knowledge\n\n"
    
    report += "### 🔧 Technical Implementation\n\n"
    report += "1. **Feature Calculation**: Pre-compute population baselines from historical real data\n"
    report += "2. **Real-time Scoring**: Apply trained model to new submissions\n"
    report += "3. **Threshold Tuning**: Adjust sensitivity based on operational needs\n"
    report += "4. **Monitoring**: Track model performance and retrain as needed\n\n"
    
    # Limitations and considerations
    report += "## Limitations and Considerations\n\n"
    report += "1. **Training Data Bias**: Model trained on specific experimental fake data\n"
    report += "2. **Generalization**: Performance may vary with different types of fake data\n"
    report += "3. **Population Drift**: Baselines may need updating as populations change\n"
    report += "4. **False Positives**: Balance between catching fakes and avoiding false accusations\n\n"
    
    report += "---\n\n"
    report += f"*Report generated by Fake Data Detection Analysis Pipeline*  \n"
    report += f"*Timestamp: {timestamp}*"
    
    return report

