#!/usr/bin/env python

import os
import subprocess

for (dirpath, dirnames, filenames) in os.walk('../processed_data'):
    for filename in filenames:
        if filename.endswith('.wav') and not filename.startswith('s_'): 
            silenced_file = os.path.join(dirpath, 's_' + filename)
            filename = os.path.join(dirpath, filename)
            bashCommand = 'sox {} {} silence 1 0.1 1% -1 0.1 1%'.format(filename, silenced_file)
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            process.communicate()
            os.remove(filename)
