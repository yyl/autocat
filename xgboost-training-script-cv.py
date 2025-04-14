#!/usr/bin/env python3
import polars as pl
import numpy as np
import argparse
import logging
import os
import sys
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import XGBoost with helpful error message
try:
    import xgboost as xgb
except ImportError as e:
    logger.error("Failed to import XGBoost. Original error: " + str(e))
    logger.error("\nPlease try the following solutions:")
    
    if sys.platform == "darwin":  # macOS
        logger.error("On macOS, install OpenMP with: brew install libomp")
    elif sys.platform == "win32":  # Windows
        logger.error("On Windows, ensure you have the Microsoft Visual C++ Redistributable installed")
    elif sys.platform.startswith("linux"):  # Linux
        logger.error("On Linux, install OpenMP with your package manager, e.g., apt install libgomp1")
    
    logger.error("\nAfter installing OpenMP, reinstall XGBoost with:")
    logger.error("pip uninstall xgboost")
    logger.error("pip install xgboost")
    
    logger.error("\nIf using Conda, try:")
    logger.error("conda install -c conda-forge xgboost")
    
    sys.exit(1)

# Now import sklearn
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

def parse_args():
    parser = argparse.ArgumentParser(description='Train XGBoost model using data from CSV, SQLite, and Parquet')
    parser.add_argument('--csv_file', required=True, help='Path to CSV file containing IDs')
    parser.add_argument('--sqlite_file', required=True, help='Path to SQLite database file')
    parser.add_argument('--table_name', required=True, help='Name of table in SQLite database')
    parser.add_argument('--parquet_file', required=True, help='Path to Parquet file containing embeddings')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--output_model', default='xgboost_model.json', help='Path to save the trained model')
    parser.add_argument('--save_label_mapping', default='label_mapping.json', help='Path to save label mapping')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_lightgbm', action='store_true', help='Use LightGBM instead of XGBoost')
    return parser.parse_args()

def load_data(csv_file, sqlite_file, table_name, parquet_file):
    """Load and merge data from CSV, SQLite and Parquet files using native Polars."""
    logger.info("Loading data from input files")
    
    # Read IDs from CSV
    try:
        ids_df = pl.read_csv(csv_file, infer_schema=False)
        if 'id' not in ids_df.columns:
            raise ValueError(f"CSV file does not contain 'id' column. Available columns: {ids_df.columns}")
        logger.info(f"Loaded {ids_df.height} IDs from CSV")
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise
    
    # Get labels from SQLite using Polars' native database support
    try:
        # Create SQLite URI
        sqlite_uri = f"sqlite://{sqlite_file}"
        
        # Get list of IDs for the query
        id_list = ids_df.select('id').to_series().to_list()
        
        # Create a parameterized query using placeholders
        placeholders = ','.join([f"'{id}'" for id in id_list])
        query = f"SELECT id, label FROM {table_name} WHERE id IN ({placeholders})"
        
        # Read from database
        labels_df = pl.read_database_uri(query=query, uri=sqlite_uri)
        
        if labels_df.height == 0:
            raise ValueError(f"No matching IDs found in SQLite table {table_name}")
        
        logger.info(f"Loaded {labels_df.height} labels from SQLite")
    except Exception as e:
        logger.error(f"Error loading from SQLite: {e}")
        raise
    
    # Get embeddings from Parquet
    try:
        parquet_df = pl.read_parquet(parquet_file)
        if 'id' not in parquet_df.columns or 'embedding64' not in parquet_df.columns:
            raise ValueError(f"Parquet file must contain 'id' and 'embedding64' columns. Available columns: {parquet_df.columns}")
        
        # Filter parquet by IDs in CSV
        parquet_df = parquet_df.filter(pl.col('id').is_in(ids_df.select('id').to_series()))
        logger.info(f"Loaded {parquet_df.height} embeddings from Parquet")
    except Exception as e:
        logger.error(f"Error loading Parquet file: {e}")
        raise
    
    # Merge data - join labels_df with parquet_df
    merged_df = labels_df.join(parquet_df, on='id', how='inner')
    logger.info(f"Final merged dataset has {merged_df.height} samples")
    
    if merged_df.height < ids_df.height:
        missing_count = ids_df.height - merged_df.height
        logger.warning(f"{missing_count} IDs from CSV were not found in both SQLite and Parquet")
    
    return merged_df

def preprocess_data(df):
    """Extract features and labels from the Polars dataframe with label encoding."""
    # Handle embeddings - convert to numpy arrays directly from Polars
    # First collect the embedding column as a Python list
    embedding_list = df.select('embedding64').to_series().to_list()
    
    # Check if embeddings are stored as strings and convert if needed
    if isinstance(embedding_list[0], str):
        embedding_list = [np.array(eval(emb)) for emb in embedding_list]
    
    # Stack embeddings into a 2D array
    X = np.stack(embedding_list)
    
    # Get original string labels
    original_labels = df.select('label').to_series().to_list()
    
    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(original_labels)
    
    # Store original IDs for reporting
    ids = df.select('id').to_numpy().flatten()
    
    # Log label mapping
    classes = label_encoder.classes_
    mapping = {i: cls for i, cls in enumerate(classes)}
    logger.info(f"Label encoding mapping: {mapping}")
    
    logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    return X, y, ids, label_encoder

def train_and_evaluate_model(X, y, ids, label_encoder, args):
    """Train model with cross-validation and evaluate."""
    logger.info(f"Performing {args.n_folds}-fold cross-validation")
    
    # Determine if binary or multi-class
    num_classes = len(np.unique(y))
    if num_classes == 2:
        logger.info("Binary classification task detected")
    else:
        logger.info(f"Multi-class classification task detected with {num_classes} classes")
    
    # Create model
    if args.use_lightgbm:
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=args.random_seed
            )
            logger.info("Using LightGBM model")
        except ImportError:
            logger.error("LightGBM is not installed. Install with: pip install lightgbm")
            logger.error("Falling back to XGBoost...")
            # Fall back to XGBoost
            model = xgb.XGBClassifier(
                objective="binary:logistic" if num_classes == 2 else "multi:softprob",
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=args.random_seed
            )
    else:
        # Use XGBoost
        model = xgb.XGBClassifier(
            objective="binary:logistic" if num_classes == 2 else "multi:softprob",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.random_seed
        )
        logger.info("Using XGBoost model")
    
    # Set up k-fold with stratification to maintain class distribution
    kf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_seed)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    
    # Get predictions for each fold for detailed reporting
    y_pred = cross_val_predict(model, X, y, cv=kf)
    
    # Print results
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"Individual fold accuracies: {', '.join([f'{score:.4f}' for score in cv_scores])}")
    
    # Generate classification report with proper class names
    report = classification_report(y, y_pred, target_names=label_encoder.classes_, output_dict=True)
    
    # Convert the dict report to a formatted string
    report_str = "Classification report:\n"
    report_str += f"{'':>20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n"
    
    # Add rows for each class
    for class_name in label_encoder.classes_:
        idx = list(label_encoder.classes_).index(class_name)
        cls_metrics = report[class_name]
        report_str += f"{class_name:>20} {cls_metrics['precision']:>10.2f} {cls_metrics['precision']:>10.2f} {cls_metrics['f1-score']:>10.2f} {cls_metrics['support']:>10.0f}\n"
    
    # Add aggregate rows
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report:
            report_str += f"{avg_type:>20} {report[avg_type]['precision']:>10.2f} {report[avg_type]['recall']:>10.2f} {report[avg_type]['f1-score']:>10.2f} {report[avg_type]['support']:>10.0f}\n"
    
    # Add accuracy
    if 'accuracy' in report:
        report_str += f"{'accuracy':>20} {report['accuracy']:>10.2f} {' ':>10} {' ':>10} {' ':>10}\n"
    
    logger.info("\n" + report_str)
    
    # Train final model on all data
    logger.info("Training final model on all data")
    model.fit(X, y)
    
    return model, label_encoder

def save_label_mapping(label_encoder, filename):
    """Save label mapping to a JSON file."""
    import json
    
    # Create mapping dictionary
    mapping = {
        'classes': label_encoder.classes_.tolist(),
        'class_to_index': {cls: idx for idx, cls in enumerate(label_encoder.classes_)},
        'index_to_class': {idx: cls for idx, cls in enumerate(label_encoder.classes_)}
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    logger.info(f"Label mapping saved to {filename}")

def main():
    args = parse_args()
    
    # Load and preprocess data
    merged_df = load_data(args.csv_file, args.sqlite_file, args.table_name, args.parquet_file)
    X, y, ids, label_encoder = preprocess_data(merged_df)
    
    # Train and evaluate model
    model, label_encoder = train_and_evaluate_model(X, y, ids, label_encoder, args)
    
    # Save model
    if hasattr(model, 'save_model'):
        model.save_model(args.output_model)
        logger.info(f"Model saved to {args.output_model}")
    else:
        # If using LightGBM or another model without save_model method
        import joblib
        joblib.dump(model, args.output_model)
        logger.info(f"Model saved to {args.output_model} using joblib")
    
    # Save label mapping
    save_label_mapping(label_encoder, args.save_label_mapping)
    
    # Save feature importance information if it's available
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Log top 10 important features
        logger.info("\nFeature importance ranking (top 10):")
        for i in range(min(10, len(indices))):
            logger.info(f"Feature {indices[i]}: {importances[indices[i]]:.4f}")

if __name__ == "__main__":
    main()