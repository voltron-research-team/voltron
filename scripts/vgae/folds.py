import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def check_fold_file_paths(fold, root_path):
    try:
        fold_label = fold.iloc[0]["fold_label"]
        random_file_path = os.path.normpath(os.path.join(root_path, fold.sample()["file_path"].values[0]))
        is_valid = os.path.exists(random_file_path)

        if is_valid:
            logger.info(f"Random file path is valid for {fold_label}. Random file path: {random_file_path}")
            return True
    except Exception as e:
        logger.error(f"An exception occurred when checking fold file paths: {e}", exc_info=True)
        return False

def get_folds(fold_index, csv_path, test=True):
    try:
        df = pd.read_csv(csv_path)
        folds = list(df.groupby('fold'))
        fold=folds[fold_index][1]
        train_fold = fold[fold["fold_label"] == "train"]

        if len(train_fold) == 0:
            raise ValueError("Train fold is empty.")
        
        if train_fold.isnull().values.any():
            raise ValueError("Null values found in train_fold")

        if test:
            test_fold = fold[fold["fold_label"] == "test"]
            
            if len(test_fold) == 0:
                raise ValueError("Test fold is empty.")
            
            if test_fold.isnull().values.any():
                raise ValueError("Null values found in test_fold")

            logger.info(f"CSV file is read from {csv_path}. Number of folds: {len(folds)}. Selected Fold Index: {fold_index}. Selected Fold: {fold.iloc[0]['fold']}. Train fold size: {len(train_fold)} and Test fold size: {len(test_fold)}")
            
            return train_fold, test_fold
        
        logger.info(f"CSV file is read from {csv_path}. Number of folds: {len(folds)}. Selected Fold Index: {fold_index}. Selected Fold: {fold.iloc[0]['fold']}. Train fold size: {len(train_fold)}")

        return train_fold, None
    except Exception as e:
        raise Exception(f"An exception occurred when getting fold: {e}")