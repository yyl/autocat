#!/usr/bin/env python3

import json
import sqlite3
import sys
import os
from typing import Dict, List, Any, Tuple

def create_corrected_table(conn: sqlite3.Connection) -> None:
    """Create the corrected table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS corrected (
        id TEXT PRIMARY KEY,
        label TEXT,
        corrected_label TEXT
    )
    ''')
    conn.commit()

def get_db_labels(conn: sqlite3.Connection, table_name: str, ids: List[str]) -> Dict[str, str]:
    """Get labels from the database for the specified IDs."""
    cursor = conn.cursor()
    
    # Validate table name to prevent SQL injection
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    if not cursor.fetchone():
        print(f"Error: Table '{table_name}' does not exist in the database.")
        sys.exit(1)
    
    # Use parameterized query for safety
    placeholders = ', '.join(['?'] * len(ids))
    query = f"SELECT id, label FROM {table_name} WHERE id IN ({placeholders})"
    
    try:
        cursor.execute(query, ids)
        return {row[0]: row[1] for row in cursor.fetchall()}
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        print(f"Failed query: {query}")
        sys.exit(1)

def process_corrections(
    conn: sqlite3.Connection, 
    json_data: List[Dict[str, Any]], 
    db_labels: Dict[str, str],
    table_name: str
) -> Tuple[int, int]:
    """
    Process the corrections and:
    1. Write them to the corrected table
    2. Update the original table with corrected labels
    
    Returns:
        Tuple of (corrections_count, updates_count)
    """
    cursor = conn.cursor()
    corrections_count = 0
    updates_count = 0
    
    for item in json_data:
        if 'id' not in item or 'label' not in item:
            continue
            
        item_id = str(item['id'])  # Ensure ID is a string
        json_label = item['label']
        
        # Skip if ID doesn't exist in database
        if item_id not in db_labels:
            continue
            
        db_label = db_labels[item_id]
        
        # Only process if labels are different
        if json_label != db_label:
            try:
                # 1. Insert into corrected table
                cursor.execute(
                    "INSERT OR REPLACE INTO corrected (id, label, corrected_label) VALUES (?, ?, ?)",
                    (item_id, db_label, json_label)
                )
                corrections_count += 1
                
                # 2. Update original table
                update_query = f"UPDATE {table_name} SET label = ? WHERE id = ?"
                cursor.execute(update_query, (json_label, item_id))
                updates_count += 1
                
            except sqlite3.Error as e:
                print(f"Error processing correction for ID {item_id}: {e}")
    
    conn.commit()
    return corrections_count, updates_count

def main():
    if len(sys.argv) != 4:
        print("Usage: python create_corrections.py <json_file> <db_file> <table_name>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    db_file = sys.argv[2]
    table_name = sys.argv[3]
    
    # Check if files exist
    if not os.path.isfile(json_file):
        print(f"Error: JSON file '{json_file}' not found.")
        sys.exit(1)
    
    if not os.path.isfile(db_file):
        print(f"Error: Database file '{db_file}' not found.")
        sys.exit(1)
    
    # Load JSON data
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            
        if not isinstance(json_data, list):
            print("Error: JSON file should contain an array of objects.")
            sys.exit(1)
            
        # Extract IDs from JSON
        json_ids = [str(item['id']) for item in json_data if 'id' in item]
        
        if not json_ids:
            print("No valid IDs found in the JSON file.")
            sys.exit(1)
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_file)
        
        # Create corrected table
        create_corrected_table(conn)
        
        # Get labels from database
        db_labels = get_db_labels(conn, table_name, json_ids)
        
        # Process corrections and updates
        corrections_count, updates_count = process_corrections(conn, json_data, db_labels, table_name)
        
        print(f"Successfully added {corrections_count} corrections to the 'corrected' table.")
        print(f"Successfully updated {updates_count} rows in the '{table_name}' table.")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()