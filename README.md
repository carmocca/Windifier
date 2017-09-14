# Windifier
Wind [instrument] Classifier. Using CNNs.
## Prerequisites
```
pip install librosa
pip install tensorflow
// Preferably tensorflow-gpu
```
## Execution

### Pre-processing steps
 (All scripts inside ./src/scripts unless otherwise specified)
1. (Optional) Convert any raw_data files to .wav using ./raw_data/to_wav.py
1. (Optional) Compress and save said files 
2. Run import_files.py to extract all the tar.gz files
3. (Optional?) Run convert_sample_rates.py to convert every files' sample rate to 44100 Hz
4. Run remove_silences.py to remove all rests from the sound files
5. Run split_files.py to split each soundfile into 1 seconds files
6. Run count_duration.py to get a sum of the file's duration for each instrument, it also deletes any files whose length is less than 1 second
7. Run generate_csv.py to generate the labels file

## Observations about papers
* Fast Fourier transform (FFT)
* Fractional Fourier transform (FRFT)
* Short-time Fourier transform (STFT)
* Mel-frequency cepstral coefficients (MFCC) (coefficients that collectively make up an Mel-frequency cepstrum, or MFC)
* "Strait mel-frequencies" (Step 2 of MFCC derivation)
* Constant-Q transform
* Spectral centroid
```
╔═══════════╦═════╦══════╦══════╦══════╦══════════╦════════════╦═══════════════════╗
║           ║ FFT ║ FRFT ║ STFT ║ MFCC ║ srait-MF ║ Constant-Q ║ Spectral Centroid ║
╠═══════════╬═════╬══════╬══════╬══════╬══════════╬════════════╬═══════════════════╣
║ Bhojane   ║     ║  X   ║      ║  X   ║          ║            ║                   ║
║ mlachmish ║     ║      ║      ║  X   ║    X     ║            ║                   ║
║ Bosch     ║     ║      ║      ║  X   ║          ║            ║                   ║
║ Pons      ║     ║      ║  X   ║  X   ║          ║            ║                   ║
║ Park      ║     ║      ║      ║  X   ║          ║            ║                   ║
║ Loughran  ║     ║      ║      ║      ║          ║            ║                   ║
║ Chetry    ║     ║      ║      ║  X   ║          ║     X      ║                   ║
║ Stowell   ║     ║      ║      ║  X   ║          ║            ║         X         ║
║ Toghiani  ║  X  ║      ║      ║  X   ║          ║            ║                   ║
╚═══════════╩═════╩══════╩══════╩══════╩══════════╩════════════╩═══════════════════╝

```

### Sources

* **Bosch**: Training 6705 audio excerpts of 3 second length labeled with a single predominant instrument. Testing split contains 2874 audio excerpts of length 5~20 seconds labeled with more than one predominant instrument. Audios samples at 44.1kHz. CNN based on small-rectangular filters (of size 3×3)
* **Pons**: Two architectures based on Bosch design strategy, Single-layer and Multi-layer. Implementation in Veleslavia/EUSIPCO2017
* **Park**: Although Timbre is decided with multiple aspects of acoustic featurs, spectrum of audio seems to be most affected feature. 13 ceptral coefficients to represents MFC, Hanning weighting window to apply before FFT, 40 Mel Filter Banks of 130 6854 Hz, 1KB block size and 512B step size.
* **Loughran**: Summary of everything, comparing different papers and techniques
* **Chetry**: Musical instrument recognition. Pag 58
