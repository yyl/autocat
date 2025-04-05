import json
from datetime import datetime
from collections import defaultdict
import sys
from decimal import Decimal, InvalidOperation

def analyze_transactions(filename):
    # Load the JSON file
    try:
        with open(filename, 'r') as f:
            transactions = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: File {filename} is not valid JSON.")
        return
    
    # Basic stats
    total_transactions = len(transactions)
    print(f"Total number of transactions: {total_transactions}")
    
    # Transactions by year
    transactions_by_year = defaultdict(int)
    missing_date = 0
    
    for transaction in transactions:
        if "date" in transaction:
            date_str = transaction["date"]
            
            try:
                # Try MM/DD/YYYY format first (your primary format)
                date_obj = datetime.strptime(date_str, "%m/%d/%Y")
                year = date_obj.year
                transactions_by_year[year] += 1
            except ValueError:
                print(f"Warning: Could not parse date format for transaction {transaction['id']}: {date_str}")
                missing_date += 1
        else:
            missing_date += 1
    
    if missing_date > 0:
        print(f"\nWarning: {missing_date} transactions had missing or invalid dates")
    
    print("\nTransactions by year:")
    for year, count in sorted(transactions_by_year.items()):
        percentage = (count / total_transactions) * 100
        print(f"{year}: {count} ({percentage:.2f}%)")
    
    # Analysis by labels
    label_counts = defaultdict(int)
    label_amounts = defaultdict(Decimal)
    total_amount = Decimal('0')
    amount_errors = 0
    
    for transaction in transactions:
        label = transaction.get("label", "unknown")
        
        # Handle amount safely, accounting for parentheses indicating negative values
        try:
            amount_str = str(transaction.get("amount", "0")).strip()
            
            # Check if amount is in parentheses (negative value)
            is_negative = amount_str.startswith('(') and amount_str.endswith(')')
            if is_negative:
                # Remove parentheses and make it negative
                amount_str = amount_str[1:-1]
                
            # Remove any currency symbols or commas
            amount_str = amount_str.replace("$", "").replace(",", "")
            
            amount = Decimal(amount_str)
            
            # Apply negative sign if needed
            if is_negative:
                amount = -amount
                
        except (InvalidOperation, ValueError, TypeError):
            print(f"Warning: Invalid amount format for transaction {transaction.get('id', 'unknown')}: '{transaction.get('amount', 'N/A')}'")
            amount_errors += 1
            amount = Decimal('0')
        
        label_counts[label] += 1
        label_amounts[label] += amount
        total_amount += amount
    
    if amount_errors > 0:
        print(f"\nWarning: {amount_errors} transactions had invalid amount formats")
    
    print("\nTransactions by label:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        count_percentage = (count / total_transactions) * 100
        print(f"{label}: {count} transactions ({count_percentage:.2f}%)")
    
    print("\nAmount by label:")
    for label, amount in sorted(label_amounts.items(), key=lambda x: x[1], reverse=True):
        amount_percentage = (amount / total_amount) * 100 if total_amount > 0 else 0
        print(f"{label}: ${amount:.2f} ({amount_percentage:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_transactions(sys.argv[1])
    else:
        print("Usage: python analyze_transactions.py [filename]")
        print("Please provide the path to the JSON file as a command-line argument.")