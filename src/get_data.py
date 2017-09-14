import numpy as np
import pandas as pd

csv_file  = './labels.csv'
csv = pd.read_csv(csv_file, header=0)
classes = ['saxophone', 'trumpet'] #  'trombone', 'clarinet'

def get_labels(index, labels=csv['label']):
    return [i for i in labels[index]]
    
def get_melspectrograms(index, paths=csv['path']):
    import log_melspectrogram as ms
    ms.set_param_values()
    spectrograms = np.asarray([ms.logscale_melspectrogram(i) for i in paths[index]], dtype=np.float32)
    return spectrograms
    
def get_mfccs(index, paths=csv['path']):
    import mfcc as mf
    mf.set_param_values()
    mfccs = np.asarray([mf.mfcc(i) for i in paths[index]], dtype=np.float32)
    return mfccs
