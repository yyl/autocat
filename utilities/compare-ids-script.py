#!/usr/bin/env python3

import csv
import json
import sys
import os
from typing import Set, Dict, List, Tuple

def read_ids_from_csv(csv_file: str) -> Set[str]:
    """Extract IDs from a CSV file and return as a set."""
    ids = set()
    try:
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if 'id' not in reader.fieldnames:
                print(f"Error: CSV file '{csv_file}' does not contain an 'id' column.")
                sys.exit(1)
            
            for row in reader:
                ids.add(row['id'])
        return ids
    except Exception as e:
        print(f"Error reading CSV file '{csv_file}': {e}")
        sys.exit(1)

def read_ids_from_json(json_file: str) -> Set[str]:
    """Extract IDs from a JSON file and return as a set."""
    ids = set()
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Handle different JSON structures
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'id' in item:
                    ids.add(str(item['id']))  # Convert to string for consistent comparison
        elif isinstance(data, dict):
            if 'id' in data:  # Single object with ID
                ids.add(str(data['id']))
            else:  # Dictionary of objects with IDs as keys or in values
                for key, value in data.items():
                    if isinstance(value, dict) and 'id' in value:
                        ids.add(str(value['id']))
                    else:
                        # Try using keys as IDs if they're not nested objects
                        ids.add(str(key))
                        
        return ids
    except Exception as e:
        print(f"Error reading JSON file '{json_file}': {e}")
        sys.exit(1)

def compare_ids(csv_ids: Set[str], json_ids: Set[str]) -> Tuple[Set[str], Set[str]]:
    """Compare two sets of IDs and return the differences."""
    missing_in_json = csv_ids - json_ids
    extra_in_json = json_ids - csv_ids
    return missing_in_json, extra_in_json

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_ids.py <csv_file> <json_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    json_file = sys.argv[2]
    
    # Check if files exist
    if not os.path.isfile(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)
    
    if not os.path.isfile(json_file):
        print(f"Error: JSON file '{json_file}' not found.")
        sys.exit(1)
    
    # Read IDs from both files
    csv_ids = read_ids_from_csv(csv_file)
    json_ids = read_ids_from_json(json_file)
    
    # Compare IDs
    missing_in_json, extra_in_json = compare_ids(csv_ids, json_ids)
    
    # Print summary
    print(f"Total IDs in CSV: {len(csv_ids)}")
    print(f"Total IDs in JSON: {len(json_ids)}")
    print(f"IDs in both files: {len(csv_ids & json_ids)}")
    
    # Print details of differences
    if missing_in_json:
        print(f"\nIDs in CSV but missing from JSON ({len(missing_in_json)}):")
        for id_value in sorted(missing_in_json):
            print(f"  - {id_value}")
    else:
        print("\nAll IDs from CSV are present in the JSON file.")
    
    if extra_in_json:
        print(f"\nIDs in JSON but not in CSV ({len(extra_in_json)}):")
        for id_value in sorted(extra_in_json):
            print(f"  - {id_value}")
    else:
        print("\nNo extra IDs in the JSON file.")
    
    # Final status
    if not missing_in_json and not extra_in_json:
        print("\nResult: The ID sets in both files match perfectly.")
    else:
        print("\nResult: There are differences between the ID sets.")

if __name__ == "__main__":
    main()
