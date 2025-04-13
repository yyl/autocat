#!/usr/bin/env python3

import csv
import sys
import os
from collections import Counter
from typing import Dict, List, Tuple

def check_csv_duplicates(csv_file: str) -> Tuple[int, int, Dict[str, int], List[str]]:
    """
    Check a CSV file for duplicate IDs.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        Tuple containing:
        - Total number of rows
        - Number of unique IDs
        - Dictionary of duplicate IDs with their counts
        - List of rows with missing IDs
    """
    all_ids = []
    missing_id_rows = []
    row_count = 0
    
    try:
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            # Check if 'id' column exists
            if 'id' not in reader.fieldnames:
                print(f"Error: CSV file '{csv_file}' does not contain an 'id' column.")
                sys.exit(1)
            
            # Process each row
            for row_num, row in enumerate(reader, start=2):  # Start at 2 to account for header
                row_count += 1
                
                # Check for empty ID
                if not row['id'] or row['id'].strip() == '':
                    missing_id_rows.append(f"Row {row_num}")
                    continue
                    
                all_ids.append(row['id'])
                
        # Count occurrences of each ID
        id_counts = Counter(all_ids)
        
        # Filter for IDs that appear more than once
        duplicates = {id_val: count for id_val, count in id_counts.items() if count > 1}
        
        return row_count, len(set(all_ids)), duplicates, missing_id_rows
        
    except Exception as e:
        print(f"Error processing CSV file '{csv_file}': {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_csv_duplicates.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.isfile(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)
    
    # Check for duplicates
    row_count, unique_count, duplicates, missing_ids = check_csv_duplicates(csv_file)
    
    # Print results
    print(f"CSV File Analysis: {csv_file}")
    print(f"Total rows: {row_count}")
    print(f"Unique IDs: {unique_count}")
    print(f"Missing IDs: {len(missing_ids)}")
    
    if duplicates:
        print("\nDuplicate IDs found:")
        print("-" * 40)
        print("ID                                     | Count")
        print("-" * 40)
        
        # Sort duplicates by count (highest first)
        for id_val, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True):
            print(f"{id_val:<40} | {count}")
            
        print(f"\nTotal duplicate IDs: {len(duplicates)}")
        print(f"Total duplicate rows: {sum(duplicates.values()) - len(duplicates)}")
    else:
        print("\nNo duplicate IDs found.")
    
    if missing_ids:
        print("\nRows with missing IDs:")
        for row in missing_ids[:10]:  # Show first 10
            print(f"  - {row}")
            
        if len(missing_ids) > 10:
            print(f"  ... and {len(missing_ids) - 10} more.")
    
    # Explain the discrepancy
    expected_unique = row_count - (sum(duplicates.values()) - len(duplicates)) - len(missing_ids)
    if expected_unique != unique_count:
        print("\nWARNING: Calculation discrepancy detected!")
    
    print(f"\nExplanation of the 400 vs {unique_count} discrepancy:")
    print(f"  - Total rows in CSV: {row_count}")
    if duplicates:
        print(f"  - Duplicate ID entries: {sum(duplicates.values()) - len(duplicates)}")
    if missing_ids:
        print(f"  - Rows with missing IDs: {len(missing_ids)}")
    print(f"  = Unique IDs: {unique_count}")

if __name__ == "__main__":
    main()
