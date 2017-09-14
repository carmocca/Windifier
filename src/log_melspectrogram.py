import sys
import numpy as np
import librosa as lb  

def set_param_values(sr=22050, n_fft=2048, n_mels=128, hop_len=512):
    global SR, N_FFT, N_MELS, HOP_LEN, REF_ROW    
    SR      = sr
    N_FFT   = n_fft
    N_MELS  = n_mels
    HOP_LEN = hop_len

def logscale_melspectrogram(path, should_plot=False):
    y, sr = lb.load(path, sr=SR)
    S = lb.logamplitude(lb.feature.melspectrogram(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS)**2)

    if should_plot:
        import matplotlib.pyplot as plt
        import librosa.display
        plt.figure()
        lb.display.specshow(S, y_axis='mel', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.show()
        print(S) # (N_MELS, ceil((SR*DUR)/HOP_LEN))
        print(S.shape)

    return S

if __name__ == '__main__':
    set_param_values()
    #set_param_values(12000, 512, 96, 256)
    logscale_melspectrogram(sys.argv[1], True)
