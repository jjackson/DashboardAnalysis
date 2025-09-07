#!/usr/bin/env python3
"""
Create a smaller, representative real data sample:
- 3 randomly selected FLWs per opportunity
- Include ALL opportunities
- All visits for selected FLWs
"""

import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os

def create_real_sample(input_file: str, output_file: str):
    """
    Create representative sample of real data.
    
    Strategy:
    1. Load all data efficiently in chunks
    2. Group by opportunity_id to see all opportunities
    3. For each opportunity, randomly select 3 FLWs
    4. Include ALL visits for selected FLWs
    """
    print("📊 CREATING REPRESENTATIVE REAL DATA SAMPLE")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # First pass: discover opportunities and FLWs
    print("\n🔍 Phase 1: Discovering opportunities and FLWs...")
    
    opportunity_flws = {}  # opp_id -> set of flw_ids
    chunk_size = 10000
    total_rows = 0
    
    for chunk_num, chunk_df in enumerate(pd.read_csv(input_file, chunksize=chunk_size), 1):
        total_rows += len(chunk_df)
        print(f"  📦 Processing chunk {chunk_num}: {len(chunk_df):,} rows (total: {total_rows:,})")
        
        for _, row in chunk_df.iterrows():
            opp_id = row['opportunity_id']
            flw_id = row['flw_id']
            
            if opp_id not in opportunity_flws:
                opportunity_flws[opp_id] = set()
            opportunity_flws[opp_id].add(flw_id)
    
    print(f"✅ Discovery complete: {total_rows:,} total rows")
    print(f"📈 Found {len(opportunity_flws)} opportunities")
    
    # Show opportunity stats
    for opp_id, flws in opportunity_flws.items():
        print(f"  Opportunity {opp_id}: {len(flws)} unique FLWs")
    
    # Phase 2: Select 3 random FLWs per opportunity
    print(f"\n🎯 Phase 2: Selecting 3 random FLWs per opportunity...")
    
    selected_flws = set()  # All selected FLW IDs across all opportunities
    selection_summary = {}
    
    np.random.seed(42)  # For reproducible sampling
    
    for opp_id, flws in opportunity_flws.items():
        flw_list = list(flws)
        
        if len(flw_list) <= 3:
            # Take all FLWs if 3 or fewer
            selected = flw_list
        else:
            # Randomly select 3
            selected = np.random.choice(flw_list, size=3, replace=False).tolist()
        
        selection_summary[opp_id] = {
            'total_flws': len(flw_list),
            'selected_flws': selected,
            'selected_count': len(selected)
        }
        
        selected_flws.update(selected)
        
        print(f"  Opportunity {opp_id}: selected {len(selected)}/{len(flw_list)} FLWs")
    
    print(f"✅ Selection complete: {len(selected_flws)} unique FLWs selected across all opportunities")
    
    # Phase 3: Extract all visits for selected FLWs
    print(f"\n📋 Phase 3: Extracting all visits for selected FLWs...")
    
    selected_records = []
    processed_rows = 0
    
    for chunk_num, chunk_df in enumerate(pd.read_csv(input_file, chunksize=chunk_size), 1):
        processed_rows += len(chunk_df)
        
        # Filter for selected FLWs
        chunk_selected = chunk_df[chunk_df['flw_id'].isin(selected_flws)]
        
        if len(chunk_selected) > 0:
            selected_records.append(chunk_selected)
            print(f"  📦 Chunk {chunk_num}: {len(chunk_selected):,}/{len(chunk_df):,} rows selected (total processed: {processed_rows:,})")
        else:
            print(f"  📦 Chunk {chunk_num}: 0/{len(chunk_df):,} rows selected (total processed: {processed_rows:,})")
    
    # Combine all selected records
    if selected_records:
        final_df = pd.concat(selected_records, ignore_index=True)
        print(f"✅ Extraction complete: {len(final_df):,} total visits selected")
        
        # Save to file
        final_df.to_csv(output_file, index=False)
        print(f"💾 Saved to: {output_file}")
        
        # Summary statistics
        print(f"\n📊 FINAL SAMPLE SUMMARY:")
        print(f"  Total visits: {len(final_df):,}")
        print(f"  Unique FLWs: {final_df['flw_id'].nunique()}")
        print(f"  Unique opportunities: {final_df['opportunity_id'].nunique()}")
        
        # Visits per opportunity
        opp_counts = final_df['opportunity_id'].value_counts().sort_index()
        print(f"\n  Visits per opportunity:")
        for opp_id, count in opp_counts.items():
            selected_flw_count = selection_summary[opp_id]['selected_count']
            print(f"    Opportunity {opp_id}: {count:,} visits from {selected_flw_count} FLWs")
        
        # Visits per FLW (top 10)
        flw_counts = final_df['flw_id'].value_counts()
        print(f"\n  Top 10 FLWs by visit count:")
        for flw_id, count in flw_counts.head(10).items():
            print(f"    FLW {flw_id}: {count:,} visits")
        
        return final_df
    else:
        print("❌ No records selected!")
        return None

def main():
    """Main function to create real data sample."""
    # Find the large real data file
    real_files = glob.glob('data/real_data_raw_*.csv')
    
    if not real_files:
        print("❌ No real data files found!")
        return
    
    input_file = max(real_files, key=lambda x: os.path.getmtime(x))
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'data/real_sample_3flws_per_opp_{timestamp}.csv'
    
    print(f"🎯 Creating representative real data sample...")
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output file: {output_file}")
    
    # Create the sample
    sample_df = create_real_sample(input_file, output_file)
    
    if sample_df is not None:
        print(f"\n🎉 SUCCESS! Representative sample created:")
        print(f"   📄 File: {output_file}")
        print(f"   📊 Size: {len(sample_df):,} visits")
        print(f"   👥 FLWs: {sample_df['flw_id'].nunique()} workers")
        print(f"   🎯 Opportunities: {sample_df['opportunity_id'].nunique()} opportunities")
        print(f"\n💡 This sample maintains the diversity of all opportunities")
        print(f"   while being manageable for analysis!")

if __name__ == "__main__":
    main()
