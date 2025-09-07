#!/usr/bin/env python3
"""
Data loading module - handles loading and preparation of fake/real datasets.
Merged from streamlined_data_loader for simplified architecture.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os
from typing import Optional, Tuple
from models.form_model_subset import TargetedFormData

def find_latest_files(data_dir="data", prefer_transformed=True) -> Tuple[Optional[str], Optional[str]]:
    """Find the most recent fake and real data files, preferring transformed versions."""
    data_path = Path(data_dir)
    
    if prefer_transformed:
        # Look for transformed files first
        fake_transformed_files = glob.glob(str(data_path / "fake_data_raw_*_transformed_*.csv"))
        real_transformed_files = glob.glob(str(data_path / "real_data_raw_*_transformed_*.csv"))
        
        if fake_transformed_files and real_transformed_files:
            fake_file = max(fake_transformed_files, key=os.path.getmtime)
            real_file = max(real_transformed_files, key=os.path.getmtime)
            return fake_file, real_file
    
    # Fallback to raw files
    fake_files = glob.glob(str(data_path / "fake_data_raw_*.csv"))
    # Exclude transformed files from raw search
    fake_files = [f for f in fake_files if "_transformed_" not in f]
    
    # Prefer full real data files over sample files
    real_sample_files = glob.glob(str(data_path / "real_sample_*.csv"))
    real_full_files = glob.glob(str(data_path / "real_data_raw_*.csv"))
    # Exclude transformed files from raw search
    real_full_files = [f for f in real_full_files if "_transformed_" not in f]

    fake_file = max(fake_files, key=os.path.getmtime) if fake_files else None

    if real_full_files:
        real_file = max(real_full_files, key=os.path.getmtime)
    elif real_sample_files:
        real_file = max(real_sample_files, key=os.path.getmtime)
    else:
        real_file = None
    
    return fake_file, real_file

def load_transformed_csv(csv_file: str, dataset_name: str, 
                        max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load pre-transformed CSV file directly (much faster than extraction).
    """
    print(f"⚡ Loading pre-transformed {dataset_name} data...")
    
    # Load CSV efficiently
    if max_samples:
        df = pd.read_csv(csv_file, nrows=max_samples)
    else:
        df = pd.read_csv(csv_file)
    
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} features")
    
    return df

def load_csv_with_extraction(csv_file: str, dataset_name: str, 
                           max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load CSV and extract the 18 target fields using the form model.
    """
    print(f"📊 Loading {dataset_name} data (with feature extraction)...")
    
    # Load CSV efficiently
    if max_samples:
        df = pd.read_csv(csv_file, nrows=max_samples)
    else:
        df = pd.read_csv(csv_file)
    
    print(f"   Processing {len(df)} records...")
    
    extracted_records = []
    valid_count = 0
    
    for i, row in df.iterrows():
        try:
            # Use the form model for extraction
            targeted_data = TargetedFormData.from_raw_record(row.to_dict())
            
            # Convert to flat dictionary
            flat_record = targeted_data.to_flat_dict()
            flat_record['data_source'] = dataset_name
            
            extracted_records.append(flat_record)
            valid_count += 1
            
        except Exception as e:
            # Skip problematic records but continue
            continue
    
    result_df = pd.DataFrame(extracted_records)
    
    print(f"✅ Successfully extracted {valid_count}/{len(df)} records")
    print(f"   Fields: {len(result_df.columns)} columns")
    
    return result_df

def load_data(max_real_samples=100000, use_transformed=True):
    """
    Load fake and real datasets for analysis.
    
    Args:
        max_real_samples: Maximum real samples to load
        use_transformed: Whether to use pre-transformed files (much faster)
        
    Returns:
        tuple: (real_df, fake_df)
    """
    print("🎯 LOADING FAKE DATA DETECTION DATASETS")
    print("=" * 50)
    
    # Find files (prefer transformed if available)
    fake_file, real_file = find_latest_files(prefer_transformed=use_transformed)
    
    if not fake_file or not real_file:
        raise FileNotFoundError("Missing required data files!")
    
    # Check if we're using transformed files
    is_transformed = "_transformed_" in fake_file and "_transformed_" in real_file
    
    if is_transformed:
        print("Using pre-transformed files for fast loading!")
    else:
        print("Using raw files - will extract 18 target fields + core metadata")
    
    print(f"📁 Using files:")
    print(f"   Fake: {Path(fake_file).name}")
    print(f"   Real: {Path(real_file).name}")
    
    # Load datasets using appropriate method
    if is_transformed:
        fake_df = load_transformed_csv(fake_file, 'fake')
        real_df = load_transformed_csv(real_file, 'real', max_samples=max_real_samples)
    else:
        fake_df = load_csv_with_extraction(fake_file, 'fake')
        real_df = load_csv_with_extraction(real_file, 'real', max_samples=max_real_samples)
    
    # Basic data validation
    print(f"\n📋 DATA SUMMARY:")
    print(f"   Real data: {len(real_df):,} visits from {real_df['flw_id'].nunique()} workers")
    print(f"   Fake data: {len(fake_df):,} visits from {fake_df['flw_id'].nunique()} workers")
    
    # Check for required fields
    required_fields = ['flw_id', 'childs_age_in_month', 'no_of_children', 'soliciter_muac_cm']
    missing_fields = []
    
    for field in required_fields:
        if field not in real_df.columns or field not in fake_df.columns:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"⚠️  Warning: Missing required fields: {missing_fields}")
    else:
        print(f"✅ All required fields present")
    
    # Show key field coverage
    key_fields = ['childs_age_in_month', 'no_of_children', 'soliciter_muac_cm', 
                  'child_name', 'household_phone']
    
    print(f"\n🎯 KEY FIELD COVERAGE:")
    for field in key_fields:
        if field in fake_df.columns and field in real_df.columns:
            fake_pct = fake_df[field].notna().sum() / len(fake_df) * 100
            real_pct = real_df[field].notna().sum() / len(real_df) * 100
            print(f"   {field:<25} Fake: {fake_pct:5.1f}%  Real: {real_pct:5.1f}%")
    
    return real_df, fake_df
