#!/usr/bin/env python3

import os
import subprocess

instruments = ['saxophone', 'trumpet'] # 'trombone', 'clarinet',  

extract_path = '../processed_data/'

# Create the output folders in case they dont exist
for instrument in instruments:
    folder = extract_path + instrument
    if not os.path.exists(folder):
        subprocess.run('mkdir ' + folder, shell=True, stdout=subprocess.PIPE)

# Extract the compressed files
for (dirpath, dirnames, filenames) in os.walk('../../raw_data'):
    for filename in filenames:
        if filename.endswith('.tar.gz'): 
            filename = os.path.join(dirpath, filename)
            outpath = None
            for instrument in instruments:
                if instrument in filename:
                    outpath = extract_path + instrument
                    break
            if outpath is None:
                continue
            bashCommand = 'tar -xf {} -C {}'.format(filename, outpath)
            print(bashCommand)
            result = subprocess.run(bashCommand, shell=True, stdout=subprocess.PIPE)
