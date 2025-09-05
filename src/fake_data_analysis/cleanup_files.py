"""
Cleanup script to remove superseded files and keep only the essential ones.
Run this to clean up the fake_data_analysis directory.
"""

import os
from datetime import datetime

def cleanup_files():
    """Remove superseded files and keep only the essential ones."""
    
    print("🧹 Cleaning up fake_data_analysis directory")
    print("="*50)
    
    # Files to delete (superseded or unrelated)
    files_to_delete = [
        'analysis.py',                    # Superseded by incremental_feature_detector.py
        'enhanced_analysis.py',           # Superseded by incremental_feature_detector.py  
        'statistical_detector.py',        # Superseded by incremental_feature_detector.py
        'user_level_analysis.py',         # Superseded by incremental_feature_detector.py
        'run_analysis.py',                # Simple runner, not needed
        'superset_example.py',            # Unrelated to fake data analysis
    ]
    
    # Files to keep (essential)
    files_to_keep = [
        'data_utils.py',                  # Essential: Data loading and preprocessing
        'feature_investigator.py',        # Essential: Validates features aren't artifacts
        'incremental_feature_detector.py', # Essential: Best analysis tool
        'cleanup_files.py'                # This script
    ]
    
    print("Files to DELETE (superseded/unrelated):")
    deleted_count = 0
    for file in files_to_delete:
        if os.path.exists(file):
            print(f"  ❌ Deleting: {file}")
            os.remove(file)
            deleted_count += 1
        else:
            print(f"  ⚪ Not found: {file}")
    
    print(f"\nFiles to KEEP (essential):")
    for file in files_to_keep:
        if os.path.exists(file):
            print(f"  ✅ Keeping: {file}")
        else:
            print(f"  ⚠️  Missing: {file}")
    
    print(f"\n🎯 Cleanup Summary:")
    print(f"  Deleted: {deleted_count} files")
    print(f"  Kept: {len([f for f in files_to_keep if os.path.exists(f)])} essential files")
    
    print(f"\n📋 Remaining Python files for raw JSON analysis:")
    print(f"  1. data_utils.py - Enhance for JSON data loading")
    print(f"  2. incremental_feature_detector.py - Primary analysis tool")  
    print(f"  3. feature_investigator.py - Validate new features")
    
    print(f"\n✅ Ready for raw JSON data integration!")

if __name__ == "__main__":
    cleanup_files()
