import csv
import json
import sqlite3
import sys
import os
from typing import List, Dict, Any

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <csv_file> <sqlite_db_file> <output_json_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    db_file = sys.argv[2]
    output_json = sys.argv[3]
    
    # Check if files exist
    if not os.path.isfile(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)
    
    if not os.path.isfile(db_file):
        print(f"Error: SQLite database file '{db_file}' not found.")
        sys.exit(1)
    
    # Read IDs from CSV
    ids = read_ids_from_csv(csv_file)
    
    if not ids:
        print("No IDs found in the CSV file.")
        sys.exit(1)
    
    # Query database and get results
    results = query_database(db_file, ids)
    
    # Write results to JSON
    write_to_json(results, output_json)
    
    print(f"Successfully extracted {len(results)} records to {output_json}")

def read_ids_from_csv(csv_file: str) -> List[str]:
    """Read IDs from the CSV file."""
    ids = []
    try:
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if 'id' not in reader.fieldnames:
                print("Error: CSV file does not contain an 'id' column.")
                sys.exit(1)
            
            for row in reader:
                ids.append(row['id'])
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    return ids

def query_database(db_file: str, ids: List[str]) -> List[Dict[str, Any]]:
    """Query the database for transactions matching the IDs."""
    results = []
    
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Using parameter substitution for SQL injection protection
        placeholders = ', '.join(['?'] * len(ids))
        query = f"""
        SELECT 
            t.id,
            t.date,
            t.provider,
            t.provider_type,
            t.amount,
            t.description,
            t.category,
            lt.label,
            lt.confidence
        FROM 
            transactions t
        LEFT JOIN 
            labeled_transactions lt ON t.id = lt.id
        WHERE 
            t.id IN ({placeholders})
        """
        
        cursor.execute(query, ids)
        
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'date': row['date'],
                'provider': row['provider'],
                'provider_type': row['provider_type'],
                'amount': row['amount'],
                'description': row['description'],
                'category': row['category'],
                'label': row['label'],
                'confidence': row['confidence']
            })
        
        conn.close()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    
    return results

def write_to_json(data: List[Dict[str, Any]], output_file: str) -> None:
    """Write the results to a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error writing to JSON file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
