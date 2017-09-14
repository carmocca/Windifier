#!/usr/bin/env python3

import os
import subprocess
import datetime

for (dirpath, dirnames, filenames) in os.walk('../processed_data'):
    for filename in filenames:
        if filename.endswith('.wav') and not filename.startswith('r'): 
            inpath = os.path.join(dirpath, filename)
            bashCommand = 'soxi -r {}'.format(inpath)
            result = subprocess.run(bashCommand, shell=True, stdout=subprocess.PIPE)
            output = int(result.stdout.decode('utf-8'))
            if (output != 44100):
                # Change the sample rate
                outpath = os.path.join(dirpath, 'r_' + filename)
                bashCommand = 'sox -V1 {} -r 44100 {}'.format(inpath, outpath)
                result = subprocess.run(bashCommand, shell=True, stdout=subprocess.PIPE)
                os.remove(inpath)

