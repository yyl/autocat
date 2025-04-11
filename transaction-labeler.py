import os
import json
import sys
import re
import subprocess
import time
from tqdm import tqdm

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

def main(input_file, model="gemma3:12b", batch_size=50, delay=0.1):
    """
    Main function to execute transaction labeling using LLM.
    
    Args:
        input_file (str): Path to the input JSON file
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
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} does not exist")
            sys.exit(1)
            
        # Load input transactions
        with open(input_file, 'r') as f:
            transactions = json.load(f)
        
        print(f"Loaded {len(transactions)} transactions from {input_file}")
        
        # Create output directory in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_dir, 'output')
        os.makedirs(output_folder, exist_ok=True)
        
        # Final output file path
        output_file = os.path.join(output_folder, 'labeled_transactions_v2.json')
        
        # Check if there's existing labeled transactions progress
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    processed_transactions = json.load(f)
                print(f"Loaded {len(processed_transactions)} previously processed transactions")
                
                # Find which transactions we need to process
                processed_ids = {t["id"] for t in processed_transactions}
                print(f"Continuing from previous run with {len(processed_ids)} transactions already processed")
                
                # Generate summary of processed transactions
                generate_summary(processed_transactions)
                return processed_transactions
            except json.JSONDecodeError:
                print("Error loading existing results file. Starting from scratch.")
        
        # Label the transactions using the integrated template
        print("\nLabeling transactions...")
        labeled_transactions = label_transactions(
            transactions, 
            LABELING_TEMPLATE, 
            model, 
            batch_size, 
            delay
        )
        
        # Save labeled transactions to the final output file
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
        print("Usage: python transaction_labeler.py <input_json_file> [model] [batch_size] [delay]")
        sys.exit(1)
    
    # Parse arguments
    input_file = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gemma3:12b"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    delay = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    
    # Execute main function
    main(input_file, model, batch_size, delay)
