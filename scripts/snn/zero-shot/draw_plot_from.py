import argparse
import pandas as pd
from test import plot_metrics, generate_latex_table
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Draw plot from data')
    parser.add_argument('--file', '-f', type=str, help='Path to metric csv file')
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.file

    if not path:
        print('Please provide path to the csv file')
        return
    
    if not path.endswith('.csv'):
        print('Please provide a csv file')
        return
    
    if not os.path.exists(path):
        print('File does not exist')

    print('Drawing plot from data in', path)

    df = pd.read_csv(path)
    
    plot_metrics(df)
    generate_latex_table(df)
    print('Plot saved to plot.png')


if __name__ == '__main__':
    main()