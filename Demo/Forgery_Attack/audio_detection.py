import numpy as np
import cv2
import csv
import pickle
import pyenf
import scipy.io.wavfile
import math
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt
import librosa
from skimage.util import img_as_float
from skimage.segmentation import slic
from scipy.stats.stats import pearsonr

# Constants for file location

folder = "Audio_Recording/"
tamperedfile = "tampered_recording.wav"
audiofile = "audio_recording.wav"
powerfile = "power_recording.wav"

tampered_filepath = folder + tamperedfile
audio_filepath = folder + audiofile
power_filepath = folder + powerfile

# ENF function parameters
fs = 1000
nfft = 8192
frame_size = 2
overlap = 0

#funtion to estimate ENF
def estimate_ENF(enf_signal, harmonics):
    enf_signal_object = pyenf.pyENF(signal0=enf_signal, fs=fs, nominal=60, harmonic_multiples=harmonics, duration=0.1,
                                    strip_index=0, frame_size_secs=frame_size, nfft=nfft, overlap_amount_secs=overlap)
    enf_spectro_strip, enf_frequency_support = enf_signal_object.compute_spectrogam_strips()
    enf_weights = enf_signal_object.compute_combining_weights_from_harmonics()
    enf_OurStripCell, enf_initial_frequency = enf_signal_object.compute_combined_spectrum(enf_spectro_strip,
                                                                                          enf_weights,
                                                                                          enf_frequency_support)
    ENF = enf_signal_object.compute_ENF_from_combined_strip(enf_OurStripCell, enf_initial_frequency)
    return ENF[:-3]

#function to compare the signal similarities
def correlation_vector(ENF_signal1, ENF_signal2, window_size, shift_size):
    correlation_ENF = []
    length_of_signal = min(len(ENF_signal1), len(ENF_signal2))
    total_windows = math.ceil(( length_of_signal - window_size + 1) / shift_size)
    rho = np.zeros((1,total_windows))
    for i in range(0,total_windows):
        enf_sig1 = ENF_signal1[i * shift_size: i * shift_size + window_size]
        enf_sig2 = ENF_signal2[i * shift_size: i * shift_size + window_size]
        enf_sig1 = np.reshape(enf_sig1, (len(enf_sig1),))
        enf_sig2 = np.reshape(enf_sig2,(len(enf_sig2), ))
        r,p = pearsonr(enf_sig1, enf_sig2)
        rho[0][i] = r
    return rho,total_windows

# Load audio reocrdings
print('Loading audio recordings . . .')
audio_rec, fs = librosa.load(audio_filepath, sr=fs)  # loading the ENF wave
tampered_rec, fs = librosa.load(tampered_filepath, sr=fs)
power_rec, fs = librosa.load(power_filepath, sr=fs) 

# Estimate ENF
print('Estimating ENF . . .')
audio_enf = estimate_ENF(audio_rec, 2)
tampered_enf = estimate_ENF(tampered_rec, 2)
power_enf = estimate_ENF(power_rec,1)

# Correlation Coefficient Analysis
window_size = 120
shift_size = 5

print('Performing correlation coefficient . . .')
rho_original,total_windows_original = correlation_vector(audio_enf, power_enf,window_size,shift_size)
rho_tampered,total_windows_tampered = correlation_vector(tampered_enf, power_enf,window_size,shift_size)
t_original = np.arange(0,total_windows_original-1,1)
t_tampered = np.arange(0,total_windows_tampered-1,1)

# Plot ENF results
fig, (audio, correlate) = plt.subplots(2, 1)
audio.plot(power_enf,'g', label='Ground Truth')
audio.plot(audio_enf,'c', label='Audio Recording')
audio.set_title("Original Audio Recording", fontsize=12)
audio.ticklabel_format(useOffset=False)
audio.set_ylabel("Freq (Hz)", fontsize=12)
audio.legend(loc="lower right")

correlate.plot(t_original,rho_original[0][1:],'g')
correlate.set_title("ENF Fluctuations compared", fontsize=12)
correlate.hlines(y=0.8, xmin=0, xmax=len(t_original), colors='r', linestyles='--', lw=2)
correlate.ticklabel_format(useOffset=False)
correlate.set_ylabel("Correlation Coefficient", fontsize=12)
correlate.set_xlabel("Number of windows", fontsize=12)

# Correlation Coefficient Results
fig, (tampered, correlate) = plt.subplots(2, 1)
tampered.plot(power_enf,'g', label='Ground Truth')
tampered.plot(tampered_enf,'r', label='Replay Attack')
tampered.set_title("Tampered Recording", fontsize=12)
tampered.ticklabel_format(useOffset=False)
tampered.set_ylabel("Freq (Hz)", fontsize=12)
tampered.legend(loc="lower right")

correlate.plot(t_tampered,rho_tampered[0][1:],'g')
correlate.set_title("ENF Fluctuations compared", fontsize=12)
correlate.hlines(y=0.8, xmin=0, xmax=len(t_tampered), colors='r', linestyles='--', lw=2)
correlate.ticklabel_format(useOffset=False)
correlate.set_ylabel("Correlation Coefficient", fontsize=12)
correlate.set_xlabel("Number of windows", fontsize=12)
plt.show()