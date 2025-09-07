#!/usr/bin/env python3
"""
Helper utility to save transformed dataframes with extracted features.
This processes raw CSV files and saves them with all 18+ extracted features
for faster loading in future analysis runs.

Uses chunked processing to handle large files (5GB+) without memory issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from models.form_model_subset import TargetedFormData
import os

def save_transformed_data_chunked(input_csv_path: str, output_suffix: str = "transformed", 
                                chunk_size: int = 1000) -> str:
    """
    Load raw CSV in chunks, extract all features using the form model, and save as transformed CSV.
    This approach handles large files (5GB+) without loading everything into memory.
    
    Args:
        input_csv_path: Path to the raw CSV file
        output_suffix: Suffix to add to the output filename
        chunk_size: Number of rows to process at a time
        
    Returns:
        str: Path to the saved transformed file
    """
    input_path = Path(input_csv_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")
    
    print(f"Processing: {input_path.name}")
    print(f"Input file size: {input_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Determine dataset type from filename
    if "fake" in input_path.name.lower():
        dataset_name = "fake"
    elif "real" in input_path.name.lower():
        dataset_name = "real"
    else:
        dataset_name = "unknown"
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{input_path.stem}_{output_suffix}_{timestamp}.csv"
    output_path = input_path.parent / output_filename
    
    print(f"Processing {dataset_name} data in chunks of {chunk_size:,} rows...")
    print(f"Output file: {output_path.name}")
    
    # Process file in chunks
    total_processed = 0
    total_valid = 0
    chunk_num = 0
    header_written = False
    
    try:
        # Read CSV in chunks
        for chunk_df in pd.read_csv(input_csv_path, chunksize=chunk_size):
            chunk_num += 1
            chunk_records = len(chunk_df)
            
            print(f"  Chunk {chunk_num}: Processing {chunk_records:,} records...")
            
            # Extract features for this chunk
            extracted_records = []
            valid_count = 0
            
            for i, row in chunk_df.iterrows():
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
            
            # Convert chunk to DataFrame
            if extracted_records:
                chunk_transformed_df = pd.DataFrame(extracted_records)
                
                # Write to CSV (append mode after first chunk)
                mode = 'w' if not header_written else 'a'
                header = not header_written
                
                chunk_transformed_df.to_csv(output_path, mode=mode, header=header, index=False)
                header_written = True
                
                total_valid += valid_count
            
            total_processed += chunk_records
            
            print(f"    Extracted {valid_count:,}/{chunk_records:,} valid records")
            print(f"    Running total: {total_valid:,}/{total_processed:,} records")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        # Clean up partial file if error occurred
        if output_path.exists():
            output_path.unlink()
        raise
    
    # Print final summary
    if output_path.exists():
        output_size = output_path.stat().st_size / (1024*1024)
        input_size = input_path.stat().st_size / (1024*1024)
        
        print(f"\n✅ Processing complete!")
        print(f"Output file size: {output_size:.1f} MB")
        print(f"Records processed: {total_processed:,}")
        print(f"Valid records extracted: {total_valid:,}")
        print(f"Success rate: {(total_valid/total_processed)*100:.1f}%")
        print(f"Compression ratio: {output_size / input_size:.2f}x")
        
        # Show feature summary
        print(f"\nFeature Summary:")
        print(f"  Core metadata: opportunity_id, flw_id, visit_date, data_source")
        print(f"  Demographics: childs_age_in_month, childs_gender, child_name, no_of_children, etc.")
        print(f"  Health/MUAC: diagnosed_with_mal_past_3_months, muac_colour, soliciter_muac_cm, etc.")
        print(f"  Vaccination: have_glasses, received_va_dose_before, received_any_vaccine, etc.")
        print(f"  Recovery: diarrhea_last_month, did_the_child_recover")
    
    return str(output_path)

def save_transformed_data(input_csv_path: str, output_suffix: str = "transformed") -> str:
    """
    Legacy function - redirects to chunked version for memory efficiency.
    """
    return save_transformed_data_chunked(input_csv_path, output_suffix, chunk_size=1000)

def process_specific_files(fake_file_path: str, real_file_path: str):
    """
    Process the specific fake and real data files provided by the user.
    Uses appropriate chunk sizes based on file size.
    
    Args:
        fake_file_path: Path to the fake data CSV
        real_file_path: Path to the real data CSV
    """
    print("=" * 60)
    print("TRANSFORMING RAW DATA FILES TO FEATURE-EXTRACTED FORMAT")
    print("Using chunked processing for memory efficiency")
    print("=" * 60)
    
    results = []
    
    # Process fake data (smaller file, smaller chunks)
    print(f"\n1. PROCESSING FAKE DATA")
    print("-" * 30)
    try:
        fake_output = save_transformed_data_chunked(fake_file_path, "transformed", chunk_size=1000)
        results.append(("Fake", fake_output))
        print(f"✅ Fake data processed successfully")
    except Exception as e:
        print(f"❌ Error processing fake data: {e}")
    
    # Process real data (large 5GB file, larger chunks for efficiency)
    print(f"\n2. PROCESSING REAL DATA (5GB file)")
    print("-" * 30)
    try:
        real_output = save_transformed_data_chunked(real_file_path, "transformed", chunk_size=50000)
        results.append(("Real", real_output))
        print(f"✅ Real data processed successfully")
    except Exception as e:
        print(f"❌ Error processing real data: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    
    if results:
        print(f"Successfully processed {len(results)} files:")
        for data_type, output_path in results:
            print(f"  {data_type}: {Path(output_path).name}")
        
        print(f"\nThese transformed files can now be loaded much faster for analysis")
        print(f"since all feature extraction has been pre-computed.")
        print(f"Memory usage was kept low by processing in chunks.")
    else:
        print("No files were successfully processed.")

if __name__ == "__main__":
    # Process the specific files requested by the user
    fake_file = r"C:\Users\Jonathan Jackson\Projects\DashboardAnalysis\src\fake_data_analysis\data\fake_data_raw_095308.csv"
    real_file = r"C:\Users\Jonathan Jackson\Projects\DashboardAnalysis\src\fake_data_analysis\data\real_data_raw_20250905_095308.csv"
    
    process_specific_files(fake_file, real_file)
