#!/usr/bin/env python3
"""
Simple Superset data retriever - executes queries and saves CSV files.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path to import from utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.superset_extractor import SupersetExtractor
import utils.sql_queries as sql_queries

def main():
    print("Superset Data Retriever")
    print("=" * 50)
    
    # Initialize extractor
    extractor = SupersetExtractor()
    
    # Check for existing partial files to resume
    import glob
    existing_real_files = glob.glob('data/real_data_raw_*.csv')
    
    if existing_real_files:
        # Use the most recent existing file
        real_file = max(existing_real_files, key=os.path.getmtime)
        print(f"🔄 Found existing file: {real_file}")
        timestamp = real_file.split('_')[-1].replace('.csv', '')
    else:
        # Create new timestamp for new files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        real_file = f'data/real_data_raw_{timestamp}.csv'
    
    # Execute fake data query (small, keep in memory) - only if no existing file
    fake_file = f'data/fake_data_raw_{timestamp}.csv'
    if not os.path.exists(fake_file):
        print("\n🎭 Retrieving fake data...")
        fake_df = extractor.execute_query(sql_queries.SQL_FAKE_DATA_PARTY)
        fake_df.to_csv(fake_file, index=False)
        print(f"✅ Fake data saved: {len(fake_df)} rows")
    else:
        print(f"\n✅ Fake data already exists: {fake_file}")
    
    # Execute real data query (large, write directly to disk)
    print("\n📊 Retrieving real data...")
    result = extractor.execute_query(sql_queries.SQL_ALL_DATA_QUERY, output_file=real_file)
    print(f"✅ Real data saved: {result['total_rows'].iloc[0]} rows")
    
    print(f"\n🎯 Complete! Files saved with timestamp: {timestamp}")

if __name__ == "__main__":
    main()