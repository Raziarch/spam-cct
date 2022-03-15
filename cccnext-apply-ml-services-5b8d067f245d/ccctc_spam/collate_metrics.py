import json
import argparse
import timeit
from glob import glob
import pandas as pd


# Get Script Arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to JSON files', action='store')
    args = parser.parse_args()
    return args


# Collate json files
def collate_json(path):
    metrics = []
    for f in glob(path + '/*.json'):
        with open(f, 'r') as fp:
            metrics.append(json.load(fp))
    print json.dumps(metrics, sort_keys=True, indent=4)
    metrics_df = pd.DataFrame(metrics)
    print metrics_df
    metrics_df.to_csv(path + '/collated_metrics.csv', index=False)

if __name__ == '__main__':
    start = timeit.default_timer()
    args = get_args()
    if args.path:
        collate_json(args.path)
    stop = timeit.default_timer()
