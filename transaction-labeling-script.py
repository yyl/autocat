#!/usr/bin/env python3
import json
import subprocess
import os
import re
import time
import sqlite3
import csv
import argparse
from tqdm import tqdm  # For progress bar

def parse_arguments():
    parser = argparse.ArgumentParser(description='Label transactions using an LLM')
    parser.add_argument('--db', required=True, help='Path to the SQLite database file')
    parser.add_argument('--table', required=True, help='Name of the table in the database')
    parser.add_argument('--model', default='mlx-community/gemma-3-12b-it-4bit', 
                        help='LLM model to use (default: mlx-community/gemma-3-12b-it-4bit)')
    parser.add_argument('--batch-size', type=int, default=50, 
                        help='Batch size for processing (default: 50)')
    parser.add_argument('--delay', type=float, default=0.01, 
                        help='Delay between API calls in seconds (default: 0.01)')
    return parser.parse_args()

# Prompt template stored directly in the script
LABELING_TEMPLATE = """You are a financial transaction categorizer. Given transaction details, assign the most appropriate category from this list: [Food, Grocery, Automotive, Shopping, Travel, Home, Payment, Health, Services, Fees, Other].

Here are some examples: 
1. 
  {
    "id": "7fef7f9b-4a08-47de-a272-f8755d30e465",
    "amount": "$238.46",
    "description": "ACH Withdrawal DISCOVER E-PAYMENT",
    "category": ""
  },
Category: Payment

2. 
  {
    "id": "04568830-3386-4bfe-aa27-009f1cc40492",
    "amount": "$14.21",
    "description": "Discount Stores",
    "category": "Household"
  },
Category: Shopping

3. 
  {
    "id": "1aaed22c-8856-479f-b413-4b0a7b44f352",
    "amount": "-95.00",
    "description": "ANNUAL MEMBERSHIP FEE",
    "category": "Fees & Adjustments"
  },
Category: Fees  

4.
  {
    "id": "595259cc-cf40-401f-aa4d-a67f9d8aa185",
    "amount": "-7.82",
    "description": "GOOGLE *SVCSSHP.8313-6",
    "category": "Personal"
  },
Category: Services

To categorize a transaction:
1. Look at all the details it has, such as the merchant name, product/service name, or even location
2. Consider the amount of the transaction
3. Think about what type of product/service this transaction indicates
4. Assign the most specific appropriate category

Now categorize this transaction:
{{TRANSACTION}}

Reply ONLY two thing: the category you choose from the fixed list, and your confidence (on a scale of 0-10) of your choice
Format your response exactly like this example: 
Category: Shopping
Confidence: 8"""

def setup_db_tables(conn):
    """Create the labeled_transactions table if it doesn't exist"""
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS labeled_transactions (
        id TEXT PRIMARY KEY,
        date TEXT,
        label TEXT,
        confidence INTEGER
    )
    ''')
    conn.commit()

def load_transactions(conn, table_name):
    """Load all transactions from the specified table"""
    print(f"Loading transactions from table {table_name}...")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    
    # Convert rows to dictionaries
    transactions = []
    for row in cursor.fetchall():
        transaction = {key: row[key] for key in row.keys()}
        transactions.append(transaction)
    
    print(f"Loaded {len(transactions)} transactions from database")
    return transactions

def get_processed_transaction_ids(conn):
    """Get IDs of transactions that have already been processed"""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM labeled_transactions")
    processed_ids = {row[0] for row in cursor.fetchall()}
    return processed_ids

# Function to get prediction and confidence from model
def get_prediction(transaction, model):
    # Create prompt from template by replacing placeholder
    transaction_json = json.dumps(transaction, indent=2)
    prompt = LABELING_TEMPLATE.replace("{{TRANSACTION}}", transaction_json)
    
    # Call LLM CLI with prompt in memory (using echo instead of cat from a file)
    try:
        cmd = f"echo '{prompt}' | llm -m {model}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        # Parse the output to extract category and confidence
        output = result.stdout.strip()
        
        # Extract category using regex
        category_match = re.search(r"Category:\s*(\w+)", output)
        if category_match:
            category = category_match.group(1)
        else:
            print(f"Warning: Could not parse category from: {output}")
            category = "Error"
        
        # Extract confidence using regex
        confidence_match = re.search(r"Confidence:\s*(\d+)", output)
        if confidence_match:
            confidence = int(confidence_match.group(1))
        else:
            print(f"Warning: Could not parse confidence from: {output}")
            confidence = 0
            
        return category, confidence
    except subprocess.CalledProcessError as e:
        print(f"Error with transaction {transaction['id']}: {e}")
        return "Error", 0

def save_labeled_transaction(conn, transaction_id, date, label, confidence):
    """Save the labeled transaction to the database"""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO labeled_transactions (id, date, label, confidence) VALUES (?, ?, ?, ?)",
        (transaction_id, date, label, confidence)
    )
    conn.commit()

def main():
    args = parse_arguments()
    
    # Connect to the database
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    
    # Set up the labeled_transactions table
    setup_db_tables(conn)
    
    # Load all transactions from the specified table
    transactions = load_transactions(conn, args.table)

    # Get IDs of already processed transactions
    processed_ids = get_processed_transaction_ids(conn)
    print(f"Found {len(processed_ids)} previously processed transactions")
    
    # Filter out already processed transactions
    transactions_to_process = [t for t in transactions if t["id"] not in processed_ids]
    print(f"Remaining transactions to process: {len(transactions_to_process)}")

    # Process transactions in batches
    total_batches = (len(transactions_to_process) + args.batch_size - 1) // args.batch_size
    processed_count = 0

    for batch_idx in range(total_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(transactions_to_process))
        batch = transactions_to_process[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx+1}/{total_batches} (transactions {start_idx+1}-{end_idx})...")
        
        # Process each transaction in the batch with progress bar
        for transaction in tqdm(batch):
            # Get category and confidence
            category, confidence = get_prediction(transaction, args.model)
            
            # Save the labeled transaction to the database
            # Assuming 'date' field exists in the transaction; if not, change to appropriate field
            date_field = transaction.get("date", transaction.get("timestamp", ""))
            save_labeled_transaction(conn, transaction["id"], date_field, category, confidence)
            
            processed_count += 1
            
            # Brief delay to avoid overwhelming the model
            time.sleep(args.delay)
        
        print(f"Completed batch {batch_idx+1}/{total_batches}")
        print(f"Progress: {processed_count}/{len(transactions_to_process)} transactions processed")

    # Get final stats for reporting
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM labeled_transactions")
    total_labeled = cursor.fetchone()[0]
    
    cursor.execute("SELECT label, COUNT(*) as count FROM labeled_transactions GROUP BY label ORDER BY count DESC")
    categories = {row[0]: row[1] for row in cursor.fetchall()}
    
    cursor.execute("SELECT AVG(confidence) FROM labeled_transactions")
    avg_confidence = cursor.fetchone()[0]
    
    # Close the database connection
    conn.close()

    print(f"\nProcessing complete! Labeled {total_labeled} transactions in total.")
    
    # Generate a quick summary of the results
    print("\nCategory distribution:")
    for category, count in categories.items():
        print(f"  {category}: {count} ({count/total_labeled*100:.1f}%)")
    
    print(f"\nAverage confidence: {avg_confidence:.1f}/10")

if __name__ == "__main__":
    main()