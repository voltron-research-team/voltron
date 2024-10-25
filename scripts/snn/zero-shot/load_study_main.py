import argparse
import optuna
from experiment import experiment_with_params

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Siamese Network")
    parser.add_argument("--study_name", '-sn', type=str, help="The name of the study", required=True)
    return parser.parse_args()

def get_best_params(study_name):
    loaded_study = optuna.load_study(study_name=study_name, storage="sqlite:///db.sqlite3")
    best_params = loaded_study.best_params
    return best_params

def main():
    args = parse_args()
    study_name = args.study_name
    best_params = get_best_params(study_name)
    experiment_with_params(best_params)


if __name__ == "__main__":
    main()