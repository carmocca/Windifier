#!/usr/bin/env python3

import os
import subprocess
import csv

def get_class(path):
    if 'saxophone' in path:
        return 0
    elif 'trumpet' in path:
        return 1
    elif 'trombone' in path:
        return 2
    elif 'clarinet' in path:
        return 3

with open('../labels.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['path', 'label'])
    for (dirpath, dirnames, filenames) in os.walk("../processed_data"):
        for filename in filenames:
            if filename.endswith('.wav'): 
                inpath = os.path.join(dirpath, filename)
                writer.writerow([inpath[1:], get_class(dirpath)])

