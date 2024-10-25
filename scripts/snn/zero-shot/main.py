from experiment import experiment_with_args, experiment_with_hyperparameter_optimization
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Siamese Network")
    parser.add_argument("--hyperparameter_optimization", '-ho', action="store_true", help="Whether to perform hyperparameter optimization")

    return parser.parse_args()

def main():
    args = parse_args()
    if args.hyperparameter_optimization:
        experiment_with_hyperparameter_optimization()
    else:
        experiment_with_args()

if __name__ == "__main__":
    main()