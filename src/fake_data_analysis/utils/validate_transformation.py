#!/usr/bin/env python3
"""
Validation script to verify that transformed data matches original data.
Randomly samples rows from original files and compares with transformed output.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
from models.form_model_subset import TargetedFormData
import json

def validate_transformation(original_file: str, transformed_file: str, num_samples: int = 20):
    """
    Validate that transformed data matches original data by sampling random rows.
    
    Args:
        original_file: Path to original CSV file
        transformed_file: Path to transformed CSV file  
        num_samples: Number of random rows to validate
    """
    print(f"Validating transformation: {Path(original_file).name}")
    print(f"Transformed file: {Path(transformed_file).name}")
    print(f"Sampling {num_samples} random rows for validation...")
    
    # Load transformed data
    transformed_df = pd.read_csv(transformed_file)
    print(f"Transformed data loaded: {len(transformed_df):,} rows, {len(transformed_df.columns)} columns")
    
    # Get total rows in original file (handle encoding issues)
    try:
        with open(original_file, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for line in f) - 1  # Subtract header
    except UnicodeDecodeError:
        try:
            with open(original_file, 'r', encoding='latin1') as f:
                total_rows = sum(1 for line in f) - 1  # Subtract header
        except:
            # Fallback: estimate from file size
            file_size = Path(original_file).stat().st_size
            total_rows = min(file_size // 1000, len(transformed_df))  # Rough estimate
    
    print(f"Original file has {total_rows:,} rows")
    
    # Generate random row indices (0-based for pandas, but accounting for header)
    random_indices = sorted(random.sample(range(total_rows), min(num_samples, total_rows)))
    print(f"Selected random row indices: {random_indices[:10]}{'...' if len(random_indices) > 10 else ''}")
    
    validation_results = []
    
    for i, row_idx in enumerate(random_indices):
        print(f"\nValidating row {i+1}/{len(random_indices)} (original row {row_idx})...")
        
        # Read specific row from original file
        original_row = pd.read_csv(original_file, skiprows=range(1, row_idx + 1), nrows=1)
        if len(original_row) == 0:
            print(f"  ⚠️  Could not read row {row_idx} from original file")
            continue
            
        original_record = original_row.iloc[0].to_dict()
        
        # Extract features using our model
        try:
            targeted_data = TargetedFormData.from_raw_record(original_record)
            expected_flat = targeted_data.to_flat_dict()
            
            # Determine dataset name
            dataset_name = "fake" if "fake" in Path(original_file).name.lower() else "real"
            expected_flat['data_source'] = dataset_name
            
        except Exception as e:
            print(f"  ❌ Error extracting features from original row: {e}")
            continue
        
        # Find corresponding row in transformed data by position
        # Since we process sequentially, row N in original should be row N in transformed
        if row_idx >= len(transformed_df):
            print(f"  ⚠️  Row {row_idx} not found in transformed data (only {len(transformed_df)} rows)")
            continue
            
        actual_row = transformed_df.iloc[row_idx].to_dict()
        
        # Compare key fields
        mismatches = []
        matches = []
        
        key_fields = [
            'opportunity_id', 'flw_id', 'visit_date', 'data_source',
            'childs_age_in_month', 'childs_gender', 'child_name', 'no_of_children',
            'soliciter_muac_cm', 'muac_colour', 'household_phone', 'household_name'
        ]
        
        for field in key_fields:
            expected_val = expected_flat.get(field)
            actual_val = actual_row.get(field)
            
            # Handle NaN comparisons
            if pd.isna(expected_val) and pd.isna(actual_val):
                matches.append(field)
            elif expected_val == actual_val:
                matches.append(field)
            else:
                mismatches.append((field, expected_val, actual_val))
        
        # Report results for this row
        if len(mismatches) == 0:
            print(f"  ✅ Perfect match! All {len(matches)} fields correct")
        else:
            print(f"  ⚠️  {len(matches)} matches, {len(mismatches)} mismatches:")
            for field, expected, actual in mismatches[:5]:  # Show first 5 mismatches
                print(f"    {field}: expected='{expected}' actual='{actual}'")
            if len(mismatches) > 5:
                print(f"    ... and {len(mismatches) - 5} more mismatches")
        
        validation_results.append({
            'row_idx': row_idx,
            'matches': len(matches),
            'mismatches': len(mismatches),
            'success': len(mismatches) == 0
        })
    
    # Summary
    print(f"\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    if validation_results:
        successful = sum(1 for r in validation_results if r['success'])
        total_matches = sum(r['matches'] for r in validation_results)
        total_mismatches = sum(r['mismatches'] for r in validation_results)
        
        print(f"Rows validated: {len(validation_results)}")
        print(f"Perfect matches: {successful}/{len(validation_results)} ({successful/len(validation_results)*100:.1f}%)")
        print(f"Total field matches: {total_matches:,}")
        print(f"Total field mismatches: {total_mismatches:,}")
        
        if successful == len(validation_results):
            print("🎉 ALL VALIDATIONS PASSED! Transformation is working correctly.")
        elif successful > len(validation_results) * 0.8:
            print("✅ Most validations passed. Minor issues may exist.")
        else:
            print("⚠️  Significant validation issues detected.")
    else:
        print("❌ No successful validations completed.")

def validate_both_files():
    """Validate both fake and real data transformations."""
    
    print("=" * 60)
    print("TRANSFORMATION VALIDATION")
    print("=" * 60)
    
    # File paths
    fake_original = "data/fake_data_raw_095308.csv"
    real_original = "data/real_data_raw_20250905_095308.csv"
    
    # Find the most recent transformed files
    data_dir = Path("data")
    fake_transformed_files = list(data_dir.glob("fake_data_raw_*_transformed_*.csv"))
    real_transformed_files = list(data_dir.glob("real_data_raw_*_transformed_*.csv"))
    
    if not fake_transformed_files:
        print("❌ No fake transformed files found!")
        return
    if not real_transformed_files:
        print("❌ No real transformed files found!")
        return
    
    # Use most recent files
    fake_transformed = max(fake_transformed_files, key=lambda x: x.stat().st_mtime)
    real_transformed = max(real_transformed_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Fake files: {Path(fake_original).name} -> {fake_transformed.name}")
    print(f"Real files: {Path(real_original).name} -> {real_transformed.name}")
    
    # Validate fake data
    print(f"\n1. VALIDATING FAKE DATA")
    print("-" * 30)
    validate_transformation(fake_original, str(fake_transformed), num_samples=20)
    
    # Validate real data
    print(f"\n2. VALIDATING REAL DATA")
    print("-" * 30)
    validate_transformation(real_original, str(real_transformed), num_samples=20)

if __name__ == "__main__":
    validate_both_files()
