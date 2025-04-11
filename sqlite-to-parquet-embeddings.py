import sqlite3
import polars as pl
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer

def generate_embeddings(texts, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    """Generate embeddings for a list of texts."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def process_database(db_path, output_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch data from transactions table
    cursor.execute("SELECT id, date, amount, description, category FROM transactions")
    rows = cursor.fetchall()
    
    # Prepare data for embedding
    ids = []
    info_strings = []
    
    for row in rows:
        transaction_id, date, amount, description, category = row
        
        # Create info string and lowercase it for consistency
        info = f"date: {date}, amount: {amount}, desc: {description}, cat: {category}".lower()
        
        ids.append(transaction_id)
        info_strings.append(info)
    
    # Generate embeddings
    print(f"Generating embeddings for {len(info_strings)} transactions...")
    embeddings = generate_embeddings(info_strings)
    
    # Close SQLite connection
    conn.close()
    
    # Convert embeddings to list of lists for Polars
    embeddings_list = [emb.tolist() for emb in embeddings]
    
    # Create a Polars DataFrame
    df = pl.DataFrame({
        "id": ids,
        "embedding": embeddings_list
    })
    
    # Save to Parquet
    df.write_parquet(output_path)
    
    print(f"Successfully processed {len(ids)} transactions and saved embeddings to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from SQLite data and save to Parquet")
    parser.add_argument("db_path", help="Path to the SQLite database file")
    parser.add_argument("--output", "-o", default="transaction_embeddings.parquet", 
                        help="Output Parquet file path (default: transaction_embeddings.parquet)")
    
    args = parser.parse_args()
    process_database(args.db_path, args.output)
