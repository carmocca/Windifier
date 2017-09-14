import sys
import numpy as np
import librosa as lb  

def set_param_values(sr=22050, n_mfcc=20, hop_len=512):
    global SR, N_MFCC, HOP_LEN
    SR      = sr
    N_MFCC = n_mfcc
    HOP_LEN = hop_len

def mfcc(path, should_plot=False):
    y, sr = lb.load(path, sr=SR)
    S = lb.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LEN)

    if should_plot:
        import matplotlib.pyplot as plt
        import librosa.display
        plt.figure()
        lb.display.specshow(S, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.show()
        print(S)
        print(S.shape) # (N_MFCC, ceil((SR*DUR)/HOP_LEN))

    return S

if __name__ == '__main__':
    set_param_values()
    mfcc(sys.argv[1], True)
