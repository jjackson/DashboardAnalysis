"""
Feature Investigation Tool
Deep dive analysis to understand WHY features are effective at discriminating real vs fake data.
Helps identify if performance is due to legitimate behavioral differences or data artifacts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from data_utils import get_analysis_datasets
from datetime import datetime

class FeatureInvestigator:
    """
    Investigates individual features to understand their discriminative power.
    """
    
    def __init__(self):
        self.investigation_results = {}
    
    def investigate_household_size(self, real_df, fake_df):
        """
        Deep investigation of household size feature that achieved 100% accuracy.
        """
        print("🔍 INVESTIGATING: Household Size Feature")
        print("="*60)
        
        investigation = {
            'feature_name': 'no_of_children (Household Size)',
            'performance': {'visit_auc': 0.999, 'worker_auc': 1.000},
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Extract household size data
        real_household = real_df['no_of_children'].dropna()
        fake_household = fake_df['no_of_children'].dropna()
        
        print(f"Real data: {len(real_household):,} non-null household size values")
        print(f"Fake data: {len(fake_household):,} non-null household size values")
        print(f"Real missing rate: {real_df['no_of_children'].isna().mean():.1%}")
        print(f"Fake missing rate: {fake_df['no_of_children'].isna().mean():.1%}")
        
        # Basic statistics
        investigation['basic_stats'] = {
            'real': {
                'count': len(real_household),
                'mean': real_household.mean(),
                'std': real_household.std(),
                'min': real_household.min(),
                'max': real_household.max(),
                'median': real_household.median(),
                'missing_rate': real_df['no_of_children'].isna().mean()
            },
            'fake': {
                'count': len(fake_household),
                'mean': fake_household.mean(),
                'std': fake_household.std(),
                'min': fake_household.min(),
                'max': fake_household.max(),
                'median': fake_household.median(),
                'missing_rate': fake_df['no_of_children'].isna().mean()
            }
        }
        
        print(f"\n📊 BASIC STATISTICS:")
        print(f"Real - Mean: {investigation['basic_stats']['real']['mean']:.2f}, Std: {investigation['basic_stats']['real']['std']:.2f}")
        print(f"     Range: {investigation['basic_stats']['real']['min']:.0f} - {investigation['basic_stats']['real']['max']:.0f}")
        print(f"Fake - Mean: {investigation['basic_stats']['fake']['mean']:.2f}, Std: {investigation['basic_stats']['fake']['std']:.2f}")
        print(f"     Range: {investigation['basic_stats']['fake']['min']:.0f} - {investigation['basic_stats']['fake']['max']:.0f}")
        
        # Value distribution analysis
        print(f"\n📈 VALUE DISTRIBUTION ANALYSIS:")
        
        real_counts = real_household.value_counts().sort_index()
        fake_counts = fake_household.value_counts().sort_index()
        
        # Show top 10 most common values for each
        print(f"\nReal data - Top 10 household sizes:")
        for value, count in real_counts.head(10).items():
            pct = (count / len(real_household)) * 100
            print(f"  {value} children: {count:,} households ({pct:.1f}%)")
        
        print(f"\nFake data - Top 10 household sizes:")
        for value, count in fake_counts.head(10).items():
            pct = (count / len(fake_household)) * 100
            print(f"  {value} children: {count:,} households ({pct:.1f}%)")
        
        # Statistical tests
        print(f"\n🧪 STATISTICAL TESTS:")
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(real_household, fake_household)
        print(f"Kolmogorov-Smirnov test:")
        print(f"  Statistic: {ks_stat:.4f}")
        print(f"  P-value: {ks_pvalue:.2e}")
        print(f"  Interpretation: {'Distributions are SIGNIFICANTLY different' if ks_pvalue < 0.001 else 'Distributions may be similar'}")
        
        # Mann-Whitney U test
        mw_stat, mw_pvalue = stats.mannwhitneyu(real_household, fake_household, alternative='two-sided')
        print(f"\nMann-Whitney U test:")
        print(f"  Statistic: {mw_stat:,.0f}")
        print(f"  P-value: {mw_pvalue:.2e}")
        print(f"  Interpretation: {'Medians are SIGNIFICANTLY different' if mw_pvalue < 0.001 else 'Medians may be similar'}")
        
        investigation['statistical_tests'] = {
            'ks_test': {'statistic': ks_stat, 'pvalue': ks_pvalue},
            'mannwhitney_test': {'statistic': mw_stat, 'pvalue': mw_pvalue}
        }
        
        # Overlap analysis
        print(f"\n🔄 OVERLAP ANALYSIS:")
        
        all_values = set(real_counts.index) | set(fake_counts.index)
        real_only = set(real_counts.index) - set(fake_counts.index)
        fake_only = set(fake_counts.index) - set(real_counts.index)
        common_values = set(real_counts.index) & set(fake_counts.index)
        
        print(f"Total unique values: {len(all_values)}")
        print(f"Values only in real data: {len(real_only)} - {sorted(real_only) if len(real_only) < 20 else f'{len(real_only)} values'}")
        print(f"Values only in fake data: {len(fake_only)} - {sorted(fake_only) if len(fake_only) < 20 else f'{len(fake_only)} values'}")
        print(f"Common values: {len(common_values)}")
        
        investigation['overlap_analysis'] = {
            'total_unique_values': len(all_values),
            'real_only_values': list(real_only),
            'fake_only_values': list(fake_only),
            'common_values': list(common_values)
        }
        
        # Suspicious patterns detection
        print(f"\n🚨 SUSPICIOUS PATTERN DETECTION:")
        
        suspicious_patterns = []
        
        # Check for impossible values
        impossible_real = real_household[(real_household < 0) | (real_household > 20)]
        impossible_fake = fake_household[(fake_household < 0) | (fake_household > 20)]
        
        if len(impossible_real) > 0:
            suspicious_patterns.append(f"Real data has {len(impossible_real)} impossible values (< 0 or > 20)")
        if len(impossible_fake) > 0:
            suspicious_patterns.append(f"Fake data has {len(impossible_fake)} impossible values (< 0 or > 20)")
        
        # Check for artificial patterns (e.g., all even numbers, all multiples of 5)
        real_even_pct = (real_household % 2 == 0).mean() * 100
        fake_even_pct = (fake_household % 2 == 0).mean() * 100
        
        if abs(real_even_pct - fake_even_pct) > 20:
            suspicious_patterns.append(f"Large difference in even numbers: Real {real_even_pct:.1f}% vs Fake {fake_even_pct:.1f}%")
        
        # Check for concentration around specific values (handle empty data)
        if len(real_household) > 0:
            real_mode = real_household.mode().iloc[0] if len(real_household.mode()) > 0 else "N/A"
            real_mode_pct = (real_household == real_mode).mean() * 100 if real_mode != "N/A" else 0
            print(f"Real data mode: {real_mode} ({real_mode_pct:.1f}% of values)")
        else:
            real_mode = "N/A"
            real_mode_pct = 0
            print(f"Real data mode: No data available")
        
        if len(fake_household) > 0:
            fake_mode = fake_household.mode().iloc[0] if len(fake_household.mode()) > 0 else "N/A"
            fake_mode_pct = (fake_household == fake_mode).mean() * 100 if fake_mode != "N/A" else 0
            print(f"Fake data mode: {fake_mode} ({fake_mode_pct:.1f}% of values)")
        else:
            fake_mode = "N/A"
            fake_mode_pct = 0
            print(f"Fake data mode: No data available")
        
        if real_mode_pct > 50 or fake_mode_pct > 50:
            suspicious_patterns.append(f"High concentration around mode values")
        
        # Check missing data patterns
        missing_diff = abs(investigation['basic_stats']['real']['missing_rate'] - investigation['basic_stats']['fake']['missing_rate'])
        if missing_diff > 0.2:
            suspicious_patterns.append(f"Large difference in missing data rates: {missing_diff:.1%}")
        
        # Check for complete missing data in one dataset
        if investigation['basic_stats']['fake']['missing_rate'] == 1.0:
            suspicious_patterns.append("🚨 CRITICAL: Fake data has 100% missing values - this is pure data leakage!")
        
        if investigation['basic_stats']['real']['missing_rate'] == 1.0:
            suspicious_patterns.append("🚨 CRITICAL: Real data has 100% missing values - this is pure data leakage!")
        
        # Check for extreme outliers
        real_max = investigation['basic_stats']['real']['max']
        if real_max > 50:  # More than 50 children is highly suspicious
            suspicious_patterns.append(f"🚨 Extreme outliers in real data: max value {real_max:.0f} (impossible household size)")
        
        # Check for data corruption indicators
        real_mean = investigation['basic_stats']['real']['mean']
        if real_mean > 100:  # Average > 100 children indicates data corruption
            suspicious_patterns.append(f"🚨 Data corruption: Average household size {real_mean:.0f} is impossible")
        
        investigation['suspicious_patterns'] = suspicious_patterns
        
        if suspicious_patterns:
            print("⚠️  SUSPICIOUS PATTERNS DETECTED:")
            for pattern in suspicious_patterns:
                print(f"  - {pattern}")
        else:
            print("✅ No obvious suspicious patterns detected")
        
        # Hypothesis about why it works
        print(f"\n💡 HYPOTHESIS - Why This Feature Works:")
        
        hypotheses = []
        
        if ks_pvalue < 0.001:
            hypotheses.append("Distributions are fundamentally different - suggests different data generation processes")
        
        if len(fake_only) > 0:
            hypotheses.append(f"Fake data contains {len(fake_only)} unique values not seen in real data")
        
        if len(real_only) > 0:
            hypotheses.append(f"Real data contains {len(real_only)} unique values not seen in fake data")
        
        mean_diff = abs(investigation['basic_stats']['real']['mean'] - investigation['basic_stats']['fake']['mean'])
        if mean_diff > 1:
            hypotheses.append(f"Large difference in average household size ({mean_diff:.1f} children)")
        
        investigation['hypotheses'] = hypotheses
        
        for i, hypothesis in enumerate(hypotheses, 1):
            print(f"  {i}. {hypothesis}")
        
        # Legitimacy assessment
        print(f"\n⚖️  LEGITIMACY ASSESSMENT:")
        
        legitimacy_score = 0
        legitimacy_reasons = []
        
        # Positive indicators (legitimate behavioral differences)
        if 0.5 < mean_diff < 3:  # Reasonable difference in household size
            legitimacy_score += 1
            legitimacy_reasons.append("✅ Reasonable difference in household size averages")
        
        if len(common_values) > len(all_values) * 0.5:  # Substantial overlap in possible values
            legitimacy_score += 1
            legitimacy_reasons.append("✅ Substantial overlap in possible values")
        
        # Negative indicators (potential artifacts)
        if len(fake_only) > 10 or len(real_only) > 10:  # Too many unique values
            legitimacy_score -= 1
            legitimacy_reasons.append("⚠️  Large number of values unique to one dataset")
        
        if investigation['basic_stats']['fake']['missing_rate'] == 0 and investigation['basic_stats']['real']['missing_rate'] > 0.1:
            legitimacy_score -= 1
            legitimacy_reasons.append("⚠️  Suspicious difference in missing data patterns")
        
        if mean_diff > 5:  # Unrealistically large difference
            legitimacy_score -= 2
            legitimacy_reasons.append("🚨 Unrealistically large difference in averages")
        
        investigation['legitimacy_assessment'] = {
            'score': legitimacy_score,
            'reasons': legitimacy_reasons
        }
        
        print(f"Legitimacy Score: {legitimacy_score}/3")
        for reason in legitimacy_reasons:
            print(f"  {reason}")
        
        if legitimacy_score >= 1:
            print(f"\n✅ CONCLUSION: Feature appears to represent LEGITIMATE behavioral differences")
        elif legitimacy_score == 0:
            print(f"\n⚠️  CONCLUSION: Feature legitimacy is UNCLEAR - needs further investigation")
        else:
            print(f"\n🚨 CONCLUSION: Feature may represent DATA ARTIFACTS rather than real behavior")
        
        return investigation
    
    def generate_investigation_report(self, investigations):
        """Generate comprehensive investigation report."""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# Feature Investigation Report
**Generated**: {timestamp}

## Investigation Overview

This report investigates WHY our detection features achieve such high performance. We analyze the underlying data patterns to determine if the discrimination comes from legitimate behavioral differences or potential data artifacts.

**Suspicion Level**: HIGH - 99.9%+ accuracy suggests potential artifacts

"""
        
        for feature_name, investigation in investigations.items():
            report += f"""
## Feature: {investigation['feature_name']}

**Performance**: Visit AUC {investigation['performance']['visit_auc']:.3f} | Worker AUC {investigation['performance']['worker_auc']:.3f}

### Basic Statistics Comparison

| Metric | Real Data | Fake Data | Difference |
|--------|-----------|-----------|------------|
| Count | {investigation['basic_stats']['real']['count']:,} | {investigation['basic_stats']['fake']['count']:,} | - |
| Mean | {investigation['basic_stats']['real']['mean']:.2f} | {investigation['basic_stats']['fake']['mean']:.2f} | {abs(investigation['basic_stats']['real']['mean'] - investigation['basic_stats']['fake']['mean']):.2f} |
| Std Dev | {investigation['basic_stats']['real']['std']:.2f} | {investigation['basic_stats']['fake']['std']:.2f} | {abs(investigation['basic_stats']['real']['std'] - investigation['basic_stats']['fake']['std']):.2f} |
| Range | {investigation['basic_stats']['real']['min']:.0f}-{investigation['basic_stats']['real']['max']:.0f} | {investigation['basic_stats']['fake']['min']:.0f}-{investigation['basic_stats']['fake']['max']:.0f} | - |
| Missing Rate | {investigation['basic_stats']['real']['missing_rate']:.1%} | {investigation['basic_stats']['fake']['missing_rate']:.1%} | {abs(investigation['basic_stats']['real']['missing_rate'] - investigation['basic_stats']['fake']['missing_rate']):.1%} |

### Statistical Tests

**Kolmogorov-Smirnov Test** (Distribution Similarity):
- Statistic: {investigation['statistical_tests']['ks_test']['statistic']:.4f}
- P-value: {investigation['statistical_tests']['ks_test']['pvalue']:.2e}
- Result: {'SIGNIFICANTLY DIFFERENT distributions' if investigation['statistical_tests']['ks_test']['pvalue'] < 0.001 else 'Similar distributions'}

**Mann-Whitney U Test** (Median Comparison):
- Statistic: {investigation['statistical_tests']['mannwhitney_test']['statistic']:,.0f}
- P-value: {investigation['statistical_tests']['mannwhitney_test']['pvalue']:.2e}
- Result: {'SIGNIFICANTLY DIFFERENT medians' if investigation['statistical_tests']['mannwhitney_test']['pvalue'] < 0.001 else 'Similar medians'}

### Value Overlap Analysis

- **Total unique values**: {investigation['overlap_analysis']['total_unique_values']}
- **Real-only values**: {len(investigation['overlap_analysis']['real_only_values'])} values
- **Fake-only values**: {len(investigation['overlap_analysis']['fake_only_values'])} values  
- **Common values**: {len(investigation['overlap_analysis']['common_values'])} values

"""
            
            if investigation['overlap_analysis']['fake_only_values']:
                report += f"**Fake-only values**: {investigation['overlap_analysis']['fake_only_values']}\n"
            
            if investigation['overlap_analysis']['real_only_values']:
                report += f"**Real-only values**: {investigation['overlap_analysis']['real_only_values']}\n"
            
            report += f"""
### Suspicious Patterns

"""
            if investigation['suspicious_patterns']:
                for pattern in investigation['suspicious_patterns']:
                    report += f"⚠️  {pattern}\n"
            else:
                report += "✅ No obvious suspicious patterns detected\n"
            
            report += f"""
### Why This Feature Works - Hypotheses

"""
            for i, hypothesis in enumerate(investigation['hypotheses'], 1):
                report += f"{i}. {hypothesis}\n"
            
            report += f"""
### Legitimacy Assessment

**Score**: {investigation['legitimacy_assessment']['score']}/3

"""
            for reason in investigation['legitimacy_assessment']['reasons']:
                report += f"{reason}\n"
            
            if investigation['legitimacy_assessment']['score'] >= 1:
                conclusion = "✅ **LEGITIMATE** - Appears to represent real behavioral differences"
            elif investigation['legitimacy_assessment']['score'] == 0:
                conclusion = "⚠️  **UNCLEAR** - Needs further investigation"
            else:
                conclusion = "🚨 **SUSPICIOUS** - May represent data artifacts"
            
            report += f"\n**Conclusion**: {conclusion}\n"
        
        report += f"""
## Overall Assessment

Based on the feature investigations, we recommend:

1. **If features are legitimate**: Proceed with confidence in the detection system
2. **If features are unclear**: Conduct additional validation with subject matter experts
3. **If features are suspicious**: Investigate data collection processes and consider alternative approaches

## Next Steps

- Validate findings with domain experts familiar with the data collection process
- Examine data collection methodology for potential systematic differences
- Consider testing on additional datasets to confirm generalizability
- Implement safeguards against overfitting to artifacts

"""
        
        return report

def main():
    """Main function to investigate features."""
    
    print("Feature Investigation Analysis")
    print("="*50)
    
    # Load data
    real_df, fake_df, combined_df = get_analysis_datasets()
    
    # Sample for analysis
    sample_size = min(len(fake_df) * 3, 6000)
    real_sample = real_df.sample(n=sample_size, random_state=42)
    
    print(f"Investigating with {len(real_sample):,} real + {len(fake_df):,} fake records")
    
    # Initialize investigator
    investigator = FeatureInvestigator()
    
    # Investigate household size (the perfect performer)
    household_investigation = investigator.investigate_household_size(real_sample, fake_df)
    
    # Store results
    investigations = {
        'household_size': household_investigation
    }
    
    # Generate report
    report = investigator.generate_investigation_report(investigations)
    
    # Save report
    import os
    os.makedirs('output', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'output/feature_investigation_report_{timestamp}.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n" + "="*60)
    print("FEATURE INVESTIGATION REPORT GENERATED")
    print("="*60)
    print(f"Report saved to: {report_path}")
    
    return investigations

if __name__ == "__main__":
    main()
