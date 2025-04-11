import os
import csv
import json
import sys
import random
import pprint
import re
import uuid
import sqlite3

def clean_text(text):
    """
    Clean text by replacing multiple whitespaces, tabs, and newlines with a single space.
    
    Args:
        text (str): Input text to clean
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return text
    return re.sub(r'\s+', ' ', str(text)).strip()

def extract_provider(filename):
    """
    Extract provider name by replacing '-' and '_' with spaces and taking the first word.
    
    Args:
        filename (str): Input filename
    
    Returns:
        str: Extracted provider name
    """
    # Replace '-' and '_' with spaces, then split and take first word
    cleaned_name = filename.replace('-', ' ').replace('_', ' ')
    return cleaned_name.split()[0]

def determine_provider_type(filename):
    """
    Determine provider type based on filename.
    
    Args:
        filename (str): Input filename
    
    Returns:
        str: Provider type ('banking' or 'credit_card')
    """
    banking_prefixes = ['Chase1515', 'Chase6992', 'Discover-checking', 'Discover-saving', 'WF']
    
    for prefix in banking_prefixes:
        if filename.startswith(prefix):
            return 'banking'
    
    return 'credit_card'

def process_csv_file(file_path, file_object_counts):
    """
    Process a single CSV file and return the extracted transactions.
    
    Args:
        file_path (str): Path to the CSV file
        file_object_counts (dict): Dictionary to track objects per file
    
    Returns:
        list: List of extracted transactions
    """
    filename = os.path.basename(file_path)
    transactions = []
    
    # Default column names for WF files
    default_headers = ['Date', 'Amount', 'Col1', 'Col2', 'Description']
    
    # Extract provider name and type
    provider = extract_provider(filename)
    provider_type = determine_provider_type(filename)
    
    # Process the CSV file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as csvfile:
        # Determine if the CSV has headers
        first_line = csvfile.readline().strip()
        csvfile.seek(0)  # Reset file pointer
        
        # Prepare the reader
        if filename.startswith('WF'):
            # For WF files, use default headers
            csvreader = csv.reader(csvfile)
            headers = default_headers
        elif ',' not in first_line:
            # Keep original headers for non-WF files without comma
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)
            csvfile.seek(0)  # Reset file pointer
        else:
            # Normal CSV with headers
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)
        
        file_transactions = []
        for row in csvreader:
            # Skip empty rows
            if not row or all(cell == '' for cell in row):
                continue
                
            # Ensure row has enough elements, pad if needed
            row = row + [''] * (len(headers) - len(row))
            
            # Create transaction dictionary
            transaction = {}
            
            # Generate unique UUID
            transaction['id'] = str(uuid.uuid4())
            
            # Create blob string parts
            blob_parts = []
            
            for i, header in enumerate(headers):
                if i >= len(row):
                    continue
                    
                # Skip columns related to reference
                if header.lower() in ['reference', 'reference number']:
                    continue
                
                # Clean the value
                cleaned_value = clean_text(row[i])
                
                # Add to blob parts if value exists
                if cleaned_value:
                    blob_parts.append(f"{header}:{cleaned_value}")
                
                # Special handling for Debit/Credit columns
                if header.lower() in ['debit', 'credit']:
                    # Only add if not zero
                    if cleaned_value != '0' and cleaned_value:
                        transaction[header] = cleaned_value
                elif cleaned_value:
                    # Add other columns with non-empty values
                    transaction[header] = cleaned_value
            
            # Add provider information and type
            transaction['Provider'] = provider
            transaction['provider_type'] = provider_type
            
            # Add blob field - all fields concatenated
            transaction['blob'] = ', '.join(blob_parts)
            
            # Only add non-empty transactions
            if transaction and len(transaction) > 1:  # More than just the id
                file_transactions.append(transaction)
        
        # Update tracking
        file_object_counts[filename] = len(file_transactions)
        transactions.extend(file_transactions)
    
    return transactions

def traverse_and_process_directory(input_folder):
    """
    Recursively traverse directory and process all CSV files.
    
    Args:
        input_folder (str): Path to the input folder
        
    Returns:
        tuple: Total number of objects processed and list of transactions
    """
    all_transactions = []
    file_object_counts = {}
    
    # Walk through all directories and subdirectories
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv') or file.endswith('.CSV'):
                file_path = os.path.join(root, file)
                try:
                    transactions = process_csv_file(file_path, file_object_counts)
                    all_transactions.extend(transactions)
                    print(f"Processed {file}: {file_object_counts[file]} transactions")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    total_objects = sum(file_object_counts.values())
    return total_objects, all_transactions, file_object_counts

def map_keys(transactions):
    """
    Map transaction keys to standardized keys.
    
    Args:
        transactions (list): List of transaction dictionaries
        
    Returns:
        list: List of transactions with mapped keys
    """
    # Define key mappings
    key_mappings = {
        "date": ["date", "transaction date", "post date", "posting date"],
        "amount": ["amount", "credit", "debit", "value", "sum", "price", "cost", "payment", "total"],
        "description": ["description", "transaction description", "details", "memo", "narration", "note", "transaction", "info"],
        "category": ["category", "transaction category", "type", "classification", "group", "tag"]
    }
    
    # Generate 3 random indices for printing (changed from 5 to 3)
    random_indices = random.sample(range(len(transactions)), min(3, len(transactions)))
    
    # Transform transactions
    transformed_transactions = []
    for idx, transaction in enumerate(transactions):
        # Create a new transaction with only the mapped keys
        new_trans = {}
        
        # Keep 'id' if it exists
        if 'id' in transaction:
            new_trans['id'] = transaction['id']
        
        # Add provider field
        if 'Provider' in transaction:
            new_trans['provider'] = transaction['Provider']
            
        # Add provider_type field
        if 'provider_type' in transaction:
            new_trans['provider_type'] = transaction['provider_type']
            
        # Add blob field
        if 'blob' in transaction:
            new_trans['blob'] = transaction['blob']
        
        # Map keys based on the defined mappings
        for fixed_key, possible_keys in key_mappings.items():
            found = False
            matched_key = None
            
            # Look for matching keys in the transaction
            for possible_key in possible_keys:
                # Case-insensitive matching
                matching_keys = [k for k in transaction.keys() if k.lower() == possible_key.lower()]
                
                if matching_keys:
                    matched_key = matching_keys[0]
                    value = transaction[matched_key]
                    
                    # Remove dollar sign from amount if present
                    if fixed_key == "amount" and isinstance(value, str):
                        value = value.replace("$", "").strip()
                    
                    new_trans[fixed_key] = value
                    found = True
                    break
            
            # If no match found, add empty string
            if not found:
                new_trans[fixed_key] = ""
        
        transformed_transactions.append(new_trans)
        
        # Print specific transactions for validation
        if idx in random_indices:
            print(f"\nOriginal Transaction (index {idx}):")
            print(json.dumps(transaction, indent=2))
            print("\nTransformed Transaction:")
            print(json.dumps(new_trans, indent=2))
            print("\nKey Mapping Details:")
            for fixed_key in key_mappings.keys():
                if fixed_key in new_trans and new_trans[fixed_key] != "":
                    # Find which possible key was matched
                    matched_from = None
                    for k in transaction.keys():
                        if fixed_key == "amount" and isinstance(transaction[k], str) and isinstance(new_trans[fixed_key], str):
                            if transaction[k].replace("$", "").strip() == new_trans[fixed_key]:
                                matched_from = k
                                break
                        elif transaction[k] == new_trans[fixed_key]:
                            matched_from = k
                            break
                    print(f"{fixed_key} (matched from '{matched_from}'): {new_trans[fixed_key]}")
                else:
                    print(f"{fixed_key}: Empty (no match found)")
    
    return transformed_transactions

def save_to_sqlite(transactions, db_file):
    """
    Save transactions to SQLite database.
    
    Args:
        transactions (list): List of transaction dictionaries
        db_file (str): Path to the SQLite database file
    """
    # Connect to SQLite database (create if it doesn't exist)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    if transactions:
        # Get all column names from the first transaction
        columns = list(transactions[0].keys())
        
        # Create column definitions, ensuring id is the primary key
        column_defs = []
        for col in columns:
            if col == 'id':
                column_defs.append(f"{col} TEXT PRIMARY KEY")
            else:
                column_defs.append(f"{col} TEXT")
        
        # Create the table
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS transactions (
            {", ".join(column_defs)}
        )
        """
        cursor.execute(create_table_sql)
        
        # Insert transactions
        print(f"\nSaving {len(transactions)} transactions to SQLite database...")
        
        # Prepare placeholders for the INSERT statement
        placeholders = ", ".join(["?" for _ in columns])
        
        # Prepare INSERT statement
        insert_sql = f"""
        INSERT OR REPLACE INTO transactions ({", ".join(columns)})
        VALUES ({placeholders})
        """
        
        # Insert transactions in batches to improve performance
        batch_size = 1000
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i+batch_size]
            batch_values = [tuple(transaction.get(col, "") for col in columns) for transaction in batch]
            cursor.executemany(insert_sql, batch_values)
            conn.commit()
            print(f"  Saved batch {i//batch_size + 1}/{(len(transactions)-1)//batch_size + 1} ({min(i+batch_size, len(transactions))}/{len(transactions)} transactions)")
    
    # Commit and close connection
    conn.commit()
    conn.close()
    print(f"Database saved to: {db_file}")

def main(input_folder):
    """
    Main function to execute the CSV processing and key mapping.
    
    Args:
        input_folder (str): Path to the input folder
    """
    try:
        # Create output directory in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_dir, 'output')
        os.makedirs(output_folder, exist_ok=True)
        
        # Step 1: Process all CSV files in the input folder recursively
        print(f"Processing CSV files in {input_folder}...")
        total_objects, all_transactions, file_object_counts = traverse_and_process_directory(input_folder)
        
        # Step 2: Save intermediate transactions to a file
        intermediate_file = os.path.join(output_folder, 'transactions.json')
        with open(intermediate_file, 'w', encoding='utf-8') as json_file:
            json.dump(all_transactions, json_file, indent=2)
        
        # Print processing statistics
        print(f"\nTotal objects processed: {total_objects}")
        print("File-wise object counts:")
        for file, count in file_object_counts.items():
            print(f"  {file}: {count} objects")
        
        # Step 3: Map keys to standardized format
        print("\nMapping transaction keys...")
        transformed_transactions = map_keys(all_transactions)
        
        # Step 4: Save transformed transactions to the final output file
        output_file = os.path.join(output_folder, 'transformed_transactions.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transformed_transactions, f, indent=2)
        
        # Step 5: Save transactions to SQLite database
        db_file = os.path.join(output_folder, 'transactions.db')
        save_to_sqlite(transformed_transactions, db_file)
        
        print(f"\nProcessing completed successfully!")
        print(f"Transformed transactions saved to: {output_file}")
        print(f"SQLite database saved to: {db_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Check if folder path is provided
    if len(sys.argv) != 2:
        print("Usage: python transaction_processor.py <input_folder_path>")
        sys.exit(1)
    
    main(sys.argv[1])