# autocat

## workflow

- download transactions in CSV
- `transaction-processor.py` cleans and converts CSV files and saves into sqlite db
- `transaction-labeling-script.py` to label every transaction using LLM
- `sqlite-to-parquet-embeddings.py` generates embeddings in a parquet file
- `sqlite-sampling-script.py` to sample training and test sets
- `utilities`
	- `csv-duplicate-checker.py` to check if sampled ids have duplicates
	- `csv-sqlite-json-exporter.py` extracts transactions for training ids for labeling check 
	- `compare-ids-script.py` to compare sampled ids and extracted ids

```
uv run sqlite-sampling-script.py --db_path output/transactions.db
uv run csv-duplicate-checker.py output/training_data.csv
uv run csv-sqlite-json-exporter.py output/training_data.csv output/transactions.db output/labeled_training.json
uv run compare-ids-script.py output/training_data.csv output/labeled_training.json
```