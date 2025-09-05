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
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Execute fake data query (small, keep in memory)
    print("\n🎭 Retrieving fake data...")
    fake_df = extractor.execute_query(sql_queries.SQL_FAKE_DATA_PARTY)
    fake_df.to_csv(f'data/fake_data_raw_{timestamp}.csv', index=False)
    print(f"✅ Fake data saved: {len(fake_df)} rows")
    
    # Execute real data query (large, write directly to disk)
    print("\n📊 Retrieving real data...")
    real_file = f'data/real_data_raw_{timestamp}.csv'
    result = extractor.execute_query(sql_queries.SQL_ALL_DATA_QUERY, output_file=real_file)
    print(f"✅ Real data saved: {result['total_rows'].iloc[0]} rows")
    
    print(f"\n🎯 Complete! Files saved with timestamp: {timestamp}")

if __name__ == "__main__":
    main()