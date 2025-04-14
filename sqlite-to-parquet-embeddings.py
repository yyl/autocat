import sqlite3
import polars as pl
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap

def generate_embeddings(texts, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    """Generate embeddings for a list of texts."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def reduce_dimensions(embeddings, n_components=64, n_neighbors=15, min_dist=0.1):
    """Reduce embedding dimensions using UMAP."""
    print(f"Reducing embeddings from {embeddings.shape[1]} to {n_components} dimensions...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

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
    
    # Reduce dimensions with UMAP
    embeddings64 = reduce_dimensions(embeddings, n_components=64)
    
    # Close SQLite connection
    conn.close()
    
    # Convert embeddings to list of lists for Polars
    embeddings_list = [emb.tolist() for emb in embeddings]
    embeddings64_list = [emb.tolist() for emb in embeddings64]
    
    # Create a Polars DataFrame
    df = pl.DataFrame({
        "id": ids,
        "embedding": embeddings_list,
        "embedding64": embeddings64_list
    })
    
    # Save to Parquet
    df.write_parquet(output_path)
    
    print(f"Successfully processed {len(ids)} transactions and saved embeddings to {output_path}")
    print(f"Original embedding dimension: {embeddings.shape[1]}, Reduced dimension: 64")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from SQLite data, reduce dimensions, and save to Parquet")
    parser.add_argument("db_path", help="Path to the SQLite database file")
    parser.add_argument("--output", "-o", default="transaction_embeddings.parquet", 
                        help="Output Parquet file path (default: transaction_embeddings.parquet)")
    parser.add_argument("--dimensions", "-d", type=int, default=64,
                        help="Number of dimensions for the reduced embeddings (default: 64)")
    parser.add_argument("--neighbors", "-n", type=int, default=15,
                        help="Number of neighbors for UMAP (default: 15)")
    parser.add_argument("--min-dist", "-m", type=float, default=0.1,
                        help="Minimum distance for UMAP (default: 0.1)")
    
    args = parser.parse_args()
    
    process_database(args.db_path, args.output)