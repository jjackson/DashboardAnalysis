# Fake Data Analysis Project

## Project Overview
This analysis aims to develop algorithms to identify fake data submissions from frontline workers (FLWs) using CommCare Connect. The project compares genuine work data against data created by workers instructed to submit fake entries.

## Data Sources

### Real Data File
- **File**: `real-sqllab_fake_party_query__20250903T175231.csv`
- **Description**: Authentic data from frontline workers performing legitimate CommCare Connect work
- **Records**: ~466K entries
- **Date**: September 3, 2025

### Fake Data File  
- **File**: `fake-sqllab_716_fake_party_query__opp_716_20250904T131013.csv`
- **Description**: Data created by frontline workers instructed to attempt deceptive submissions
- **Records**: ~2K entries  
- **Date**: September 4, 2025 (most recent)

## Data Structure
Both datasets contain child health screening data with the following key fields:
- **Identity**: opportunity_id, flw_id, flw_name (fake only)
- **Temporal**: visit_date, visit_start_time, visit_end_time (fake only)
- **Demographics**: childs_age_in_month, childs_gender, child_name
- **Health Metrics**: muac_colour, soliciter_muac_cm, diagnosed_with_mal_past_3_months
- **Treatments**: under_treatment_for_mal, received_va_dose_before, received_any_vaccine
- **Household**: household_name, household_phone, no_of_children
- **Consent/Completion**: muac_consent, va_consent, all_six_no

## Key Differences Observed
1. **Column Count**: Fake data has additional columns (flw_name, visit_start_time, visit_end_time, muac_consent)
2. **Data Volume**: Real data significantly larger (~466K vs ~2K records)
3. **Temporal Patterns**: Fake data includes precise timing information

## Analysis Goals

### Two-Level Detection Strategy
1. **Individual Visit Detection**: Flag suspicious individual form submissions
   - Identify specific visits that appear fake based on data patterns
   - Each record gets a probability score for being fake
   - Useful for real-time quality control and targeted review

2. **User-Level Detection**: Identify workers submitting consistently suspicious data
   - Aggregate individual visit scores per FLW (frontline worker)
   - Detect workers with patterns indicating systematic fake data submission
   - All workers in fake dataset are fake, all workers in real dataset are legitimate
   - Useful for worker performance management and training needs

### Implementation Approach
3. **Algorithmic Development**: Create automated detection methods for both levels
4. **Dashboard Creation**: Present findings with drill-down from user-level to individual visits

## Methodology
- Compare statistical distributions between datasets
- ~~Analyze temporal patterns and submission behaviors~~ **NOTE: Temporal analysis should be IGNORED - fake data was created on single day vs real service delivery over months**
- Examine data quality indicators and anomalies
- Develop machine learning models for classification

## Data Handling Approach

### Recommended: Enhanced DataFrames
The project uses **pandas DataFrames** with structured utility functions rather than formal data models:

**Core Files (Post-Cleanup):**
- `data_utils.py`: Clean data loading and preprocessing utilities (ready for JSON enhancement)
- `incremental_feature_detector.py`: Primary analysis tool with comprehensive feature testing
- `feature_investigator.py`: Critical validation tool to prevent data leakage artifacts

**Benefits:**
- **Rapid iteration**: Perfect for exploratory analysis
- **Flexibility**: Easy schema changes as patterns emerge
- **Rich ecosystem**: Full pandas/numpy analysis capabilities
- **Memory efficient**: Handles large datasets without ORM overhead

**Key Functions:**
- `get_analysis_datasets()`: Load standardized real/fake datasets
- `get_feature_summary()`: Comprehensive statistical summaries
- `get_temporal_patterns()`: Time-based behavior analysis
- `detect_outliers()`: Automated anomaly detection

This approach prioritizes analysis speed and flexibility over formal data modeling, ideal for research and pattern discovery.

## Top 10 Detection Analysis Ideas

### 1. **Statistical Distribution Analysis**
- Compare distributions of numeric fields (age, MUAC measurements, household size)
- Use Kolmogorov-Smirnov tests, Anderson-Darling tests for distribution differences
- Detect unusual clustering or artificial patterns in continuous variables

### 2. **Categorical Pattern Detection**
- Analyze frequency distributions of categorical variables (gender, MUAC color, diagnoses)
- Chi-square tests for independence between variables
- Look for unrealistic correlations or missing expected medical patterns

### 3. **Data Quality Indicators**
- Missing data patterns (systematic vs random missingness)
- Impossible or medically inconsistent combinations
- Outlier detection using IQR, Z-scores, or Isolation Forest methods

### 4. **Text Pattern Analysis**
- Name similarity detection (repeated or generated names)
- Phone number patterns (sequential, repeated digits, invalid formats)
- Household name consistency and realism

### 5. **Medical Logic Validation**
- MUAC measurements vs malnutrition diagnosis consistency
- Age-appropriate vaccine status validation
- Treatment status vs diagnosis alignment
- Physiologically impossible combinations

### 6. **User Behavior Profiling** ⚠️ **REVISED**
- Individual FLW data quality consistency patterns
- ~~Volume and variance analysis per worker~~ **EXCLUDED: Volume differences are artifacts**
- Detect workers with suspiciously uniform or random **content patterns** (not volume)

### 7. **Benford's Law Analysis**
- Apply Benford's Law to numeric fields (ages, measurements, counts)
- Detect artificial number generation vs natural data collection
- Particularly effective for MUAC measurements and child counts

### 8. **Clustering and Anomaly Detection**
- Unsupervised clustering to identify distinct submission groups
- Isolation Forest, Local Outlier Factor, One-Class SVM
- DBSCAN for density-based anomaly detection

### 9. **Machine Learning Classification**
- Random Forest, Gradient Boosting for feature importance ranking
- Logistic Regression for interpretable probability scores
- Neural networks for complex pattern detection
- Cross-validation with proper train/test splits

### 10. **Ensemble Methods and Meta-Analysis**
- Combine multiple detection techniques into ensemble scores
- Weighted voting systems based on technique reliability
- Meta-features derived from multiple analysis results

## Analysis Framework Design

### Core Framework Structure
```
DetectionFramework
├── DataLoader (data_utils.py)
├── FeatureEngineers
│   ├── StatisticalFeatures
│   ├── MedicalLogicFeatures  
│   ├── TextPatternFeatures
│   └── BehavioralFeatures
├── DetectionMethods
│   ├── StatisticalTests
│   ├── AnomalyDetectors
│   ├── MLClassifiers
│   └── RuleBasedDetectors
├── EvaluationMetrics
│   ├── SensitivitySpecificity
│   ├── ROCAnalysis
│   ├── PrecisionRecall
│   └── ConfusionMatrix
└── EnsembleScoring
```

### Evaluation Methodology
- **Ground Truth**: Real data (label=0), Fake data (label=1)
- **Metrics**: Sensitivity, Specificity, PPV, NPV, AUC-ROC, AUC-PR
- **Thresholds**: Multiple cutoff analysis for operational flexibility
- **Cross-Validation**: Stratified K-fold to handle class imbalance
- **Feature Importance**: SHAP values for model interpretability

### Statistical Packages to Leverage
- **scipy.stats**: Statistical tests (KS, Anderson-Darling, Chi-square)
- **scikit-learn**: ML algorithms, metrics, preprocessing
- **imbalanced-learn**: Handling class imbalance (SMOTE, etc.)
- **benfordslaw**: Benford's Law analysis implementation
- **shap**: Model explainability and feature importance
- **yellowbrick**: Visualization for model evaluation
- **optuna**: Hyperparameter optimization
- **pandas-profiling**: Automated EDA and data quality reports

### Implementation Strategy
1. **Baseline Models**: Start with simple statistical tests and rule-based detection
2. **Feature Engineering**: Create comprehensive feature sets from each analysis idea
3. **Individual Method Evaluation**: Test each detection method independently
4. **Ensemble Development**: Combine best-performing methods
5. **Threshold Optimization**: Find optimal cutoffs for operational requirements
6. **Validation**: Test on held-out data and cross-validation

### Expected Challenges
- **Class Imbalance**: ~466K real vs ~2K fake records
- **Domain Expertise**: Medical logic validation requires healthcare knowledge
- **Overfitting**: Risk of learning temporal artifacts despite exclusion
- **Interpretability**: Need explainable results for operational deployment

## Standard Reporting Template

### Detection Method Report Structure
Each detection method will be documented using this standardized format:

**1. Method Overview** (2-3 sentences)
- Brief description of the detection approach
- What patterns it's designed to identify

**2. Real-World Interpretation** (2-3 sentences)  
- What the method reveals about fake vs real data collection behavior
- Why these patterns indicate fraudulent activity

**3. Performance Summary Table**
```
Detection Level    | Metric        | Threshold | Performance
-------------------|---------------|-----------|-------------
Individual Visits  | Sensitivity   | 0.3/0.5/0.7 | XX.X%
                  | Specificity   |           | XX.X%
                  | AUC-ROC       | -         | X.XXX
User-Level        | Sensitivity   | 0.3/0.5/0.7 | XX.X%
                  | Specificity   |           | XX.X%  
                  | AUC-ROC       | -         | X.XXX
```

**4. Key Findings** (3-4 bullet points)
- Most important discriminative features
- Operational recommendations
- Limitations and considerations

**Target Length**: Maximum 1 page per method

**Output Location**: All detection method reports are saved to `output/` directory for organized storage and easy access.

**Timestamping**: 
- Report filenames include timestamp: `report_name_YYYYMMDD_HHMMSS.md`
- Report content includes generation timestamp in header for easy reference within the document

## Analysis Notes & Learnings

### Key Constraints Discovered

**Cannot Use Temporal Features:**
- Fake data was created on single day vs real service delivery over months
- Any time-based patterns would be artifacts of data collection timing, not genuine behavioral differences

**Cannot Use Volume/Frequency Features:**
- Fake workers intentionally submitted much less data than real workers
- Volume differences (e.g., visits per worker, submission frequency) reflect data collection constraints, not detection signals
- Must focus on **content quality** rather than **quantity patterns**

### Valid Detection Signals Identified

**Statistical Distribution Patterns (Confirmed Effective):**
- MUAC measurement distributions show genuine behavioral differences
- Fake workers generate measurements with unnatural statistical patterns
- Real health workers show natural variation that fake workers cannot replicate
- Focus on **how measurements are distributed**, not **how many measurements**

### Analysis Evolution

**Initial Hypothesis**: Volume and temporal patterns would be strong indicators
**Revised Understanding**: Content quality and measurement patterns are the true discriminators
**Implication**: Detection methods must focus on **data authenticity** rather than **data quantity**
