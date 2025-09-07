# Fake Data Analysis Project

## Project Overview

This analysis develops algorithms to identify fake data submissions from frontline workers (FLWs) using CommCare Connect. The basis for the data we will be analyzing
is a small dataset generated where Dimagi paid workers specifically to create fake data, incentivizing them to create the most reaalistic fake data with a bonus for 
being in the top 25% of "realistic looking" data.  The large real dataset is generated from CommCare Connect workers doing service delivery in the field.

### Detection Objectives
1. **Primary Detection**: Distinguish between known/paid fake workers (30) and legitimate workers
2. **Secondary Detection**: Identify potential fradulant activity within the real worker population

### Context & Incentive Structure

**Fake Data Generation (30 workers):**
- Small pool instructed to submit fake visits with bonus incentives for "better faked data"
- Expected to have more complete data due to controlled environment and time to craft realistic submissions

**Real Data (Production workers):**
- Workers paid per visit → financial incentive to inflate volume
- Real workers who fake visits submit forms without actually visiting children
- Expected to have more missing/incomplete data due to field conditions and time pressure

### Key Constraints and Learning
**❌ Cannot Use Temporal Features:** Fake data created on single day vs real service delivery over months - any time patterns are artifacts
**❌ Cannot Use Volume/Frequency Features:** Volume differences reflect data collection constraints of the much shorter fake data generation, not detection signals
**✅ Focus on Content Quality:** Statistical distribution patterns and data authenticity are the true discriminators
**✅ Focus on Real-World Meaning of classifiers:** Successful classifiers should make real world sense as to why they have discriminatory value

## Data Architecture & Loading

### Current Implementation Status
- ✅ **18 Core Fields Identified**: Successfully extracted across both real and fake JSON data
- ✅ **Robust Data Pipeline**: JSON → Pydantic → DataFrames with memory optimization

### Data Sources
- **Fake Data**: `fake_data_raw_*.csv` (~2K records, complete JSON forms)
- **Real Data**: `real_data_raw_*.csv` (~440K records, 5GB dataset)
- **Sample Files**: `real_sample_*.csv` (preferred for development using the real data as its much smaller)

### Three-Layer Processing Pipeline

**Layer 1: Raw Data**
- Raw JSON form data with nested structure
- Core fields: opportunity_id, flw_id, flw_name, visit_date, form_json

**Layer 2: Structured Extraction (`form_model_subset.py`)**
- `TargetedFormData`: Pydantic model with robust JSON parsing
- Multiple path fallbacks handle app version differences that cause JSON structure differences
- Type-safe conversion (int, float, str) with error handling

**Layer 3: Analysis DataFrames (`streamlined_data_loader.py`)**
- Memory-efficient loading with sampling control (`max_real_samples`)
- Automatic file discovery and quality reporting
- Graceful degradation for problematic records

### 18 Analysis Fields
*Demographics & Household (7 fields):*
- `childs_age_in_month`, `childs_gender`, `child_name`, `no_of_children`
- `hh_have_children`, `household_phone`, `household_name`

*Health & MUAC (4 fields):*
- `diagnosed_with_mal_past_3_months`, `muac_colour`, `under_treatment_for_mal`, `soliciter_muac_cm`

*Vaccination & Health (5 fields):*
- `have_glasses`, `received_va_dose_before`, `received_any_vaccine`, `va_child_unwell_today`, `recent_va_dose`

*Recovery (2 fields):*
- `diarrhea_last_month`, `did_the_child_recover`

---

## Analysis Framework

### Two-Level Detection Strategy

**1. Individual Visit Detection**
- Flag suspicious individual form submissions
- Each record gets probability score for being fake
- Focus on content quality patterns fake workers cannot replicate

**2. User-Level Detection**
- Aggregate individual visit scores per FLW
- Detect workers with systematic fake data submission patterns
- Useful for worker performance management

### Core Detection Ideas

**1. Statistical Distribution Analysis**
- Compare distributions of numeric fields (age, MUAC, household size)
- Kolmogorov-Smirnov tests, Anderson-Darling tests
- Detect artificial patterns vs natural variation

**2. Medical Logic Validation**
- MUAC measurements vs malnutrition diagnosis consistency
- Age-appropriate vaccine status validation
- Physiologically impossible combinations

**3. Categorical Pattern Detection**
- Frequency distributions of categorical variables
- Chi-square tests for independence
- Unrealistic correlations or missing expected patterns

**4. Data Quality Indicators**
- Missing data patterns (systematic vs random)
- Impossible or medically inconsistent combinations
- Outlier detection (IQR, Z-scores, Isolation Forest)

**5. Text Pattern Analysis**
- Name similarity detection (repeated/generated names)
- Phone number patterns (sequential, invalid formats)
- Household information consistency

**6. Benford's Law Analysis**
- Apply to numeric fields (ages, measurements, counts)
- Detect artificial number generation vs natural collection
- Particularly effective for MUAC measurements

**7. Clustering & Anomaly Detection**
- Unsupervised clustering for distinct submission groups
- Isolation Forest, Local Outlier Factor, One-Class SVM
- DBSCAN for density-based anomaly detection

**8. Machine Learning Classification**
- Random Forest, Gradient Boosting for feature importance
- Logistic Regression for interpretable probability scores
- Cross-validation with proper train/test splits

**9. User Behavior Profiling**
- Individual FLW data quality consistency patterns
- Detect suspiciously uniform or random content patterns (not volume)

**10. Ensemble Methods**
- Combine multiple detection techniques
- Weighted voting based on technique reliability
- Meta-features from multiple analysis results

### Analysis Philosophy
- **Interpretable Features**: Prioritize explainable results over black-box performance
- **Logical Validation**: Ensure detected patterns align with expected fake visit behaviors
- **Content Over Quantity**: Focus on data authenticity rather than volume patterns
- **Production-Ready Features**: Features should be calculable using only individual worker's data, not requiring access to entire dataset for deployment

### Production-Ready Feature Design Principles

**✅ GOOD: Worker-Level Features**
- Calculate statistics within each worker's own submission history
- Compare worker's patterns to established population baselines (pre-computed)
- Use worker-specific consistency checks and medical logic validation
- Example: "Worker X has 15% name similarity within their own submissions vs 3% baseline"

**❌ AVOID: Dataset-Wide Dependencies**
- Features requiring real-time access to all other workers' data
- Cross-worker comparisons that need the full dataset
- Features that change based on who else is in the analysis batch
- Example: "Worker X's names are most similar to Worker Y's names"

**🎯 IMPLEMENTATION STRATEGY**
1. **Training Phase**: Use full dataset to establish population baselines and thresholds
2. **Production Phase**: Apply pre-computed baselines to individual worker data streams
3. **Worker Profiling**: Build features from worker's own submission patterns and consistency
4. **Medical Logic**: Use established medical knowledge, not dataset-derived patterns

**📊 FEATURE CATEGORIES**
- **Individual Consistency**: Medical logic, field completeness, value ranges within worker
- **Population Comparison**: Worker's patterns vs pre-established population statistics  
- **Temporal Patterns**: Worker's submission quality over time (content-based, not volume)
- **Content Quality**: Data authenticity indicators that don't require cross-worker comparison

---

## Reporting Framework
### Standard Detection Method Report Structure

Each detection method documented with:

**1. Method Overview** (2-3 sentences)
- Detection approach description
- Patterns designed to identify

**2. Real-World Interpretation** (2-3 sentences)
- What method reveals about fake vs real data collection
- Why patterns indicate fraudulent activity

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

### Evaluation Methodology
- **Ground Truth**: Real data (label=0), Fake data (label=1)
- **Metrics**: Sensitivity, Specificity, PPV, NPV, AUC-ROC, AUC-PR

### Output Standards
- **Location**: All reports saved to `output/` directory
- **Timestamping**: Filenames include `YYYYMMDD_HHMMSS` format
- **Target Length**: Short and Concise as feasible
- **Generation Timestamp**: Include in report header

## Current Implementation Status

### ✅ Completed
- **Data Pipeline**: `streamlined_data_loader.py` working with 18-field extraction
- **JSON Processing**: `form_model_subset.py` handles complex parsing with fallbacks
- **Field Validation**: 18 core fields successfully extract from both datasets

### 🔧 In Progress
- **Statistical Methods**: `incremental_feature_detector.py` needs alignment with 18-field model

### 📋 Next Steps
1. **Fix Column Mismatches**: Align field names between data model and analysis code
2. **Implement Detection Methods**: Build the 10 core detection approaches
3. **Two-Level Analysis**: Support both visit-level and worker-level detection
4. **Validation Pipeline**: Test methods against known fake/real patterns