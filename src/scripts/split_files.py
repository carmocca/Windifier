#!/usr/bin/env python3

import os
import subprocess

# Extract the compressed files
for (dirpath, dirnames, filenames) in os.walk("../processed_data"):
    for filename in filenames:
        if filename.endswith('.wav'): 
            filename = os.path.join(dirpath, filename)
            bashCommand = 'sox -V1 {} {} trim 0 1 : newfile : restart'.format(filename, filename)
            result = subprocess.run(bashCommand, shell=True, stdout=subprocess.PIPE)
            os.remove(filename)
