
import re
import os
import argparse
import timeit
import ast
import json


# Get Script Arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parse', help='Log file (including Path) from run', type=argparse.FileType('rt'))
    args = parser.parse_args()
    return args


# Extract breakdown of function run times
def extract_metrics(log):
    metrics = {}
    prefix, _ = os.path.splitext(log.name)
    logname = prefix + '_metrics.json'
    print logname
    for line in log:
        match = re.search('\[(.*): (.*): (.* .*)\] INFO: run_metrics: (.*) - (.*)', line)
        if match:
            metric_name = match.group(4).lower()
            metric_value = match.group(5).lower()
            # print metric_name, metric_value
            if metric_name == 'model_params':
                metrics.update(ast.literal_eval(metric_value))
            elif metric_name == 'train_performance':
                m_val = ast.literal_eval(metric_value)
                for k in m_val.keys():
                    metrics['train_' + k] = m_val[k]
            elif metric_name == 'test_performance':
                m_val = ast.literal_eval(metric_value)
                for k in m_val.keys():
                    metrics['test_' + k] = m_val[k]
            elif metric_name == 'feature_importances':
                for i, v in enumerate(ast.literal_eval(metric_value)):
                    metrics['feature_' + str(i)] = v
            else:
                metrics[metric_name] = metric_value

    with open(logname, 'wb') as fp:
        json.dump(metrics, fp, sort_keys=True, indent=0)
        fp.write('\n')
        
    print metrics

if __name__ == '__main__':
    start = timeit.default_timer()
    args = get_args()
    if args.parse:
        extract_metrics(args.parse)

    stop = timeit.default_timer()
