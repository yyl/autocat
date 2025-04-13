import sqlite3
import polars as pl
import random
import os
import argparse
from datetime import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sample records from SQLite database by year and create train/test splits')
    
    parser.add_argument('--db_path', required=True, 
                        help='Path to the SQLite database file')
    
    parser.add_argument('--output_dir', default='output',
                        help='Directory to save output CSV files (default: output)')
    
    parser.add_argument('--table_name', default='transactions',
                        help='Name of the table to query (default: transactions)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Sample sizes as specified
    samples = {
        '2024': 320,
        '2023': 160,
        '2022': 20
    }
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    try:
        # Connect to the database
        print(f"Connecting to database: {args.db_path}")
        conn = sqlite3.connect(args.db_path)
        cursor = conn.cursor()
        
        # Create a set to track sampled IDs across all years (to ensure uniqueness)
        all_sampled_ids = set()
        all_data = []
        columns = None
        
        # Process each year
        for year, sample_size in samples.items():
            print(f"Sampling {sample_size} unique records from {year}...")
            
            # Format the date condition
            start_date = f"01/01/{year}"
            end_date = f"12/31/{year}"
            
            # Get all IDs from the specified year (excluding already sampled IDs)
            if all_sampled_ids:
                placeholders = ', '.join(['?'] * len(all_sampled_ids))
                query = f"""
                SELECT id FROM {args.table_name}
                WHERE date >= ? AND date <= ?
                AND id NOT IN ({placeholders})
                """
                cursor.execute(query, (start_date, end_date, *all_sampled_ids))
            else:
                query = f"""
                SELECT id FROM {args.table_name}
                WHERE date >= ? AND date <= ?
                """
                cursor.execute(query, (start_date, end_date))
            
            # Fetch all available IDs for the year
            available_ids = [row[0] for row in cursor.fetchall()]
            
            # Check if we have enough records
            if len(available_ids) < sample_size:
                print(f"Warning: Only {len(available_ids)} unique records available for {year}, using all of them")
                sampled_ids = available_ids
            else:
                # Randomly sample unique IDs for this year
                sampled_ids = random.sample(available_ids, sample_size)
            
            # Add these IDs to our tracking set
            all_sampled_ids.update(sampled_ids)
            
            # Get full records for the sampled IDs
            if sampled_ids:
                placeholders = ', '.join(['?'] * len(sampled_ids))
                records_query = f"""
                SELECT * FROM {args.table_name}
                WHERE id IN ({placeholders})
                """
                
                cursor.execute(records_query, sampled_ids)
                
                if columns is None:
                    columns = [description[0] for description in cursor.description]
                
                year_records = cursor.fetchall()
                all_data.extend(year_records)
                print(f"Retrieved {len(year_records)} records for {year}")
        
        # Convert to Polars DataFrame
        df = pl.DataFrame(all_data, schema=columns)
        
        # Verify we have no duplicate IDs
        unique_count = df.select(pl.col("id")).unique().height
        if unique_count != len(df):
            print(f"Warning: Found {len(df) - unique_count} duplicate IDs in the final dataset")
        else:
            print(f"Success: All {unique_count} IDs in the final dataset are unique")
        
        # Shuffle the data
        df = df.sample(fraction=1.0, seed=args.seed)
        
        # Split into training (80%) and test (20%)
        split_index = int(len(df) * 0.8)
        train_df = df.slice(0, split_index)
        test_df = df.slice(split_index, len(df) - split_index)
        
        # Save to CSV files
        train_file = os.path.join(args.output_dir, "training_data.csv")
        test_file = os.path.join(args.output_dir, "test_data.csv")
        
        train_df.write_csv(train_file)
        test_df.write_csv(test_file)
        
        print(f"Saved {len(train_df)} records to {train_file}")
        print(f"Saved {len(test_df)} records to {test_file}")
        
        # Summary of the split
        total_sampled = len(df)
        total_requested = sum(samples.values())
        print("\nSummary:")
        print(f"Total records sampled: {total_sampled} out of {total_requested} requested")
        
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