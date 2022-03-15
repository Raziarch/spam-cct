
import re
import argparse
import timeit
from datetime import datetime

# Local Packages
from lib.utils import support_functions as sf


# Get Script Arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parse', help='Log file (including Path) from run', type=argparse.FileType('rt'))
    args = parser.parse_args()
    return args


# Extract breakdown of function run times
def extract_runsplits(log):
    stack = []
    log_d = {}
    level = 0
    max_level = 0
    for line in log:
        match = re.search('\[(.*): (.*): (.* .*)\] INFO: Function (.*)', line)
        if match:
            file_name = match.group(1)
            func_name = match.group(2)
            func_state = match.group(4)
            timestamp = datetime.strptime(match.group(3), '%Y-%m-%d %H:%M:%S,%f')
            print file_name, func_name, timestamp, func_state
            # Push into stack
            if func_state == 'Start':
                level += 1
                if level > max_level:
                    max_level = level
                func_string = '        '*(level - 1) + func_name
                log_d[timestamp] = {'level': level, 'func_name': func_string, 'delta': ''}
                stack.append([func_name, timestamp])
            else:
                [_, timestamp_start] = stack.pop()
                delta = timestamp - timestamp_start
                func_string = '        '*(level - 1) + func_name
                log_d[timestamp] = {'level': level, 'func_name': func_string, 'delta': delta}
                level -= 1
                # print func_name, delta

    for ts in sorted(log_d):
        try:
            cumm_delta = ts - start_tstmp
        except NameError:
            start_tstmp = ts
            cumm_delta = ''
        log_d[ts]['cumm_delta'] = cumm_delta
        # print ts, '{0:>15}'.format(cumm_delta), '{0:>15}'.format(log_d[ts]['delta']), log_d[ts]['func_name']

    # print log_d
    print('Run Split Summary: \n{}\n'.format(sf.tabulate_dict(log_d,
                                                              multilevel=True,
                                                              transpose=True,
                                                              preserve_whitespace=True)))


if __name__ == '__main__':
    start = timeit.default_timer()
    args = get_args()
    if args.parse:
        extract_runsplits(args.parse)
    stop = timeit.default_timer()
    print('Total Execution Time: %s', stop - start)
