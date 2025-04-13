import sqlite3
import polars as pl
import random
import os
import argparse
from datetime import datetime

def get_data_by_year(conn, year, sample_size):
    """Query the database for records from a specific year and randomly sample them"""
    # Format the date condition
    start_date = f"01/01/{year}"
    end_date = f"12/31/{year}"
    
    # Get all IDs from the specified year
    query = f"""
    SELECT id FROM transactions
    WHERE date >= ? AND date <= ?
    """
    
    cursor = conn.cursor()
    cursor.execute(query, (start_date, end_date))
    
    # Fetch all IDs for the year
    all_ids = [row[0] for row in cursor.fetchall()]
    
    # Randomly sample if we have more records than needed
    if len(all_ids) > sample_size:
        sampled_ids = random.sample(all_ids, sample_size)
    else:
        print(f"Warning: Only {len(all_ids)} records available for {year}, using all of them")
        sampled_ids = all_ids
    
    # Get records for the sampled IDs
    placeholders = ', '.join(['?'] * len(sampled_ids))
    records_query = f"""
    SELECT id, date FROM transactions
    WHERE id IN ({placeholders})
    """
    
    cursor.execute(records_query, sampled_ids)
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    return rows, columns

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Sample data from SQLite database and split into training and test sets')
    parser.add_argument('--db_path', type=str, required=True, help='Path to SQLite database file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output CSV files')
    
    args = parser.parse_args()
    
    # Get command line arguments
    DB_PATH = args.db_path
    OUTPUT_DIR = args.output_dir
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Sample sizes as specified
    samples = {
        '2024': 320,
        '2023': 160,
        '2022': 20
    }
    
    # Connect to the database
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Lists to hold all sampled data and columns
        all_data = []
        columns = None
        
        # Get samples for each year
        for year, sample_size in samples.items():
            print(f"Sampling {sample_size} records from {year}...")
            records, cols = get_data_by_year(conn, year, sample_size)
            
            if columns is None:
                columns = cols
            
            all_data.extend(records)
            print(f"Retrieved {len(records)} records for {year}")
        
        # Convert to Polars DataFrame
        df = pl.DataFrame(all_data, schema=columns)
        
        # Shuffle the data (Polars equivalent of sample frac=1)
        df = df.sample(fraction=1.0, seed=42)
        
        # Split into training (80%) and test (20%)
        split_index = int(len(df) * 0.8)
        train_df = df.slice(0, split_index)
        test_df = df.slice(split_index, len(df) - split_index)
        
        # Save to CSV files
        train_file = os.path.join(OUTPUT_DIR, "training_data.csv")
        test_file = os.path.join(OUTPUT_DIR, "test_data.csv")
        
        train_df.write_csv(train_file)
        test_df.write_csv(test_file)
        
        print(f"Saved {len(train_df)} records to {train_file}")
        print(f"Saved {len(test_df)} records to {test_file}")
        
        # Summary of the split
        total_samples = sum(samples.values())
        print("\nSummary:")
        print(f"Total records sampled: {len(df)} out of {total_samples} requested")
        
        # Count records by year using Polars expressions
        for year in samples:
            count = df.filter(pl.col("date").str.contains(f"/{year}")).height
            print(f"  {year}: {count} records")
            
        print(f"Training set: {len(train_df)} records")
        print(f"Test set: {len(test_df)} records")
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()