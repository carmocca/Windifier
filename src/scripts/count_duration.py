#!/usr/bin/env python3

import os
import subprocess
import datetime

instruments = {}
for (dirpath, dirnames, filenames) in os.walk("../processed_data"):
    for filename in filenames:
        if filename.endswith('.wav'): 
            filename = os.path.join(dirpath, filename)
            bashCommand = 'soxi -D {}'.format(filename)
            result = subprocess.run(bashCommand, shell=True, stdout=subprocess.PIPE)
            output = float(result.stdout.decode('utf-8'))
            # Remove files whose length is less than 1 second, usually the tail of each set of splitted files
            if output < 1:
                print(filename, output)
                os.remove(filename)
                continue

    # The number of files in each folder is the total length for that instrument since each file is 1 second long
    if len(filenames) > 0 and all(filename.endswith('.wav') for filename in filenames):
        if dirpath in instruments:
            instruments[dirpath] += len(filenames)
        else:
            instruments[dirpath] = len(filenames)

# Output the total time for each instrument
instruments.update((x, str(datetime.timedelta(seconds=y))) for x, y in instruments.items())
print(instruments)

''' 
INITIAL OUTPUT. These files gave errors because they are empty.
./saxophone/s_saxophone_F5_phrase_mezzo-forte_staccatissimo.wav
./saxophone/s_saxophone_F4_025_pianissimo_normal.wav
./saxophone/s_saxophone_Cs4_025_fortissimo_normal.wav
./saxophone/s_saxophone_As5_025_piano_normal.wav
./saxophone/s_saxophone_C5_phrase_mezzo-forte_staccatissimo.wav
./saxophone/s_saxophone_Gs3_025_fortissimo_normal.wav
./trumpet/s_trumpet_G3_025_pianissimo_normal.wav
./trumpet/s_trumpet_E4_long_pianissimo_normal.wav
./trumpet/s_trumpet_D4_long_pianissimo_normal.wav
./trumpet/s_trumpet_G3_long_piano_normal.wav
./trumpet/s_trumpet_As3_long_pianissimo_normal.wav
./trumpet/s_trumpet_A3_very-long_piano_normal.wav
./trumpet/s_trumpet_Ds5_long_pianissimo_normal.wav
./trumpet/s_trumpet_G4_025_forte_normal.wav
./trumpet/s_trumpet_D4_long_piano_normal.wav
./trumpet/s_trumpet_Ds4_025_pianissimo_normal.wav
./trumpet/s_trumpet_G3_long_mezzo-forte_normal.wav
./trumpet/s_trumpet_A4_long_pianissimo_normal.wav
./trumpet/s_trumpet_F4_long_piano_normal.wav
./trumpet/s_trumpet_G3_long_pianissimo_normal.wav
{'./saxophone': '7:55:45.007060', './trumpet': '6:46:35.233472'}


{'./saxophone': '7:50:17', './trumpet': '6:42:31'}
'''
