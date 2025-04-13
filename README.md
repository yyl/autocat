# autocat

## workflow

- download transactions in CSV
- `transaction-processor.py` cleans and converts CSV files and saves into sqlite db
- `transaction-labeling-script.py` to label every transaction using LLM
- `sqlite-to-parquet-embeddings.py` generates embeddings in a parquet file
- `sqlite-sampling-script.py` to sample training and test sets