import os
import csv
import json
import sys
import random
import pprint
import re
import uuid
import subprocess
import time
from tqdm import tqdm

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
    
    # Extract provider name
    provider = extract_provider(filename)
    
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
            
            for i, header in enumerate(headers):
                if i >= len(row):
                    continue
                    
                # Skip columns related to reference
                if header.lower() in ['reference', 'reference number']:
                    continue
                
                # Clean the value
                cleaned_value = clean_text(row[i])
                
                # Special handling for Debit/Credit columns
                if header.lower() in ['debit', 'credit']:
                    # Only add if not zero
                    if cleaned_value != '0' and cleaned_value:
                        transaction[header] = cleaned_value
                elif cleaned_value:
                    # Add other columns with non-empty values
                    transaction[header] = cleaned_value
            
            # Add provider information
            transaction['Provider'] = provider
            
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
    
    # Generate 5 random indices for printing
    random_indices = random.sample(range(len(transactions)), min(5, len(transactions)))
    
    # Transform transactions
    transformed_transactions = []
    for idx, transaction in enumerate(transactions):
        # Create a new transaction with only the mapped keys
        new_trans = {}
        
        # Keep 'id' if it exists
        if 'id' in transaction:
            new_trans['id'] = transaction['id']
        
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

def get_prediction(transaction, template, model="gemma3:12b"):
    """
    Get category prediction and confidence from an LLM without using temporary files.
    
    Args:
        transaction (dict): Transaction to categorize
        template (str): Prompt template to use
        model (str): LLM model to use
        
    Returns:
        tuple: (category, confidence)
    """
    # Create prompt from template by replacing placeholder
    transaction_json = json.dumps(transaction, indent=2)
    prompt = template.replace("{{TRANSACTION}}", transaction_json)
    
    # Call LLM CLI using echo and pipe instead of temporary file
    try:
        result = subprocess.run(
            f"echo '{prompt}' | llm -m {model}",
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

def label_transactions(transactions, template, model="gemma3:12b", batch_size=50, delay=0.1):
    """
    Label transactions using an LLM.
    
    Args:
        transactions (list): List of transactions to label
        template (str): Template string to use for prompts
        model (str): LLM model to use
        batch_size (int): Number of transactions to process in each batch
        delay (float): Delay between API calls in seconds
        
    Returns:
        list: List of labeled transactions
    """
    # Process transactions in batches
    labeled_transactions = []
    total_batches = (len(transactions) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(transactions))
        batch = transactions[start_idx:end_idx]
        
        print(f"\nLabeling batch {batch_idx+1}/{total_batches} (transactions {start_idx+1}-{end_idx})...")
        
        # Process each transaction in the batch with progress bar
        for transaction in tqdm(batch):
            # Get category and confidence
            category, confidence = get_prediction(transaction, template, model)
            
            # Add category and confidence to transaction
            transaction_copy = transaction.copy()
            transaction_copy["label"] = category
            transaction_copy["confidence"] = confidence
            labeled_transactions.append(transaction_copy)
            
            # Brief delay to avoid overwhelming the model
            time.sleep(delay)
        
        print(f"Completed batch {batch_idx+1}/{total_batches}")
        print(f"Progress: {len(labeled_transactions)}/{len(transactions)} transactions labeled")
    
    return labeled_transactions

def generate_summary(transactions):
    """
    Generate a summary of transaction labels and confidence.
    
    Args:
        transactions (list): List of labeled transactions
    """
    if not transactions:
        return
    
    categories = {}
    confidence_sum = 0
    confidence_count = 0
    
    for t in transactions:
        if "label" in t:
            categories[t["label"]] = categories.get(t["label"], 0) + 1
        if "confidence" in t:
            confidence_sum += t["confidence"]
            confidence_count += 1
    
    print("\nCategory distribution:")
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} ({count/len(transactions)*100:.1f}%)")
    
    if confidence_count > 0:
        print(f"\nAverage confidence: {confidence_sum/confidence_count:.1f}/10")

def main(input_folder, model="gemma3:12b", batch_size=50, delay=0.1):
    """
    Main function to execute the CSV processing, key mapping, and transaction labeling.
    
    Args:
        input_folder (str): Path to the input folder
        model (str): LLM model to use
        batch_size (int): Number of transactions to process in each batch
        delay (float): Delay between API calls in seconds
    """
    # Integrated labeling template as a string constant
    LABELING_TEMPLATE = """You are a financial transaction categorizer. Given transaction details, assign the most appropriate category from this list: [Food, Grocery, Shopping, Travel, Automotive, Home, Payment, Health, Services, Fees, Other].

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
1. Look at the details such as the name of the merchant, product or service in the description
2. Consider the amount of the transaction
3. Think about what type of goods or services this merchant likely provides, what kind of product this likely is
4. Assign the most specific appropriate category from the given list

Now categorize this transaction:
{{TRANSACTION}}

Reply ONLY two thing: the category you choose from the given list, and your confidence (on a scale of 0-10) of the choice
Format your response exactly like this example: 
Category: Shopping
Confidence: 8"""

    try:
        # Create output directory in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_dir, 'output')
        os.makedirs(output_folder, exist_ok=True)
        
        # Final output files path
        transformed_file = os.path.join(output_folder, 'transformed_transactions.json')
        output_file = os.path.join(output_folder, 'labeled_transactions.json')
        
        # Check if there's existing labeled transactions progress
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    processed_transactions = json.load(f)
                print(f"Loaded {len(processed_transactions)} previously processed transactions")
                
                # Find which transactions we need to process
                processed_ids = {t["id"] for t in processed_transactions}
                print(f"Continuing from previous run with {len(processed_ids)} transactions already processed")
                
                # Return the processed transactions
                generate_summary(processed_transactions)
                return processed_transactions
            except json.JSONDecodeError:
                print("Error loading existing results file. Starting from scratch.")
        
        # Check if there's existing transformed transactions
        if os.path.exists(transformed_file):
            try:
                with open(transformed_file, 'r') as f:
                    transformed_transactions = json.load(f)
                print(f"Loaded {len(transformed_transactions)} previously transformed transactions")
                # Skip the CSV processing and key mapping steps
            except json.JSONDecodeError:
                print("Error loading existing transformed file. Starting from scratch.")
                transformed_transactions = None
        else:
            transformed_transactions = None
        
        # If no existing transformed transactions, process from CSV files
        if transformed_transactions is None:
            # Step 1: Process all CSV files in the input folder recursively
            print(f"Processing CSV files in {input_folder}...")
            total_objects, all_transactions, file_object_counts = traverse_and_process_directory(input_folder)
            
            # Print processing statistics
            print(f"\nTotal objects processed: {total_objects}")
            print("File-wise object counts:")
            for file, count in file_object_counts.items():
                print(f"  {file}: {count} objects")
            
            # Step 2: Map keys to standardized format
            print("\nMapping transaction keys...")
            transformed_transactions = map_keys(all_transactions)
            
            # Step 3: Save transformed transactions to a file
            with open(transformed_file, 'w', encoding='utf-8') as f:
                json.dump(transformed_transactions, f, indent=2)
            
            print(f"Transformed transactions saved to: {transformed_file}")
        
        # Step 4: Label the transactions using the integrated template
        print("\nLabeling transactions...")
        labeled_transactions = label_transactions(
            transformed_transactions, 
            LABELING_TEMPLATE, 
            model, 
            batch_size, 
            delay
        )
        
        # Step 5: Save labeled transactions to the final output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(labeled_transactions, f, indent=2)
        
        print(f"\nProcessing completed successfully!")
        print(f"Labeled transactions saved to: {output_file}")
        
        # Generate summary
        generate_summary(labeled_transactions)
        
        return labeled_transactions
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python transaction_processor.py <input_folder_path> [model] [batch_size] [delay]")
        sys.exit(1)
    
    # Parse arguments
    input_folder = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gemma3:12b"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    delay = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    
    # Execute main function
    main(input_folder, model, batch_size, delay)