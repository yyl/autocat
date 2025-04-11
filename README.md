# autocat

## workflow

- download transactions in CSV
- `transaction-processor.py` cleans and converts CSV files and saves into sqlite db
- `sqlite-to-parquet-embeddings.py` generates embeddings in a parquet file