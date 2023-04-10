# Author: Ruilin Zhang
# Description: Authenticate audio recording with ground truth and flag attack to database

from Google import Create_Service
from googleapiclient.http import MediaFileUpload
import numpy as np
import cv2
import pickle
import pyenf
import ntplib
import datetime
import time
import scipy.io.wavfile
import math
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt
import librosa
from io import StringIO
from skimage.util import img_as_float
from skimage.segmentation import slic
from scipy.stats.stats import pearsonr

# Create google drive service instance
CLIENT_SECRET_FILE = 'credentials.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']
file_id = '1mtrbGpd94c51ZjwLZNGfv7kaTcAw5JtD'
dev = 'Laptop'

# Constants for file location
local_file = "updated_log.txt"
folder = "Audio_Recording/"
audiofile = "tampered_recording.wav"
powerfile = "power_recording.wav"

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
    attack_threshold = 10
    attack_flag = 0
    cnt = 0
    for i in range(0,total_windows):
        enf_sig1 = ENF_signal1[i * shift_size: i * shift_size + window_size]
        enf_sig2 = ENF_signal2[i * shift_size: i * shift_size + window_size]
        enf_sig1 = np.reshape(enf_sig1, (len(enf_sig1),))
        enf_sig2 = np.reshape(enf_sig2,(len(enf_sig2), ))
        r,p = pearsonr(enf_sig1, enf_sig2)
        rho[0][i] = r
        # Detect for ENF mismatch
        if (r < 0.8):
            cnt = cnt + 1
        elif cnt >= attack_threshold:
            attack_flag = 1
        else:
            cnt = 0
    return rho,total_windows, attack_flag

def UTC_check():
    ntpc = ntplib.NTPClient()
    host = 'pool.ntp.org'
    UTC_ref = time.time()

    ## Synchronize with NTP
    while(True): 
        try:
            UTC_ref = ntpc.request(host).tx_time

        except ntplib.NTPException:
            print(f'Ntp_Exception thrown, Last:{datetime.datetime.utcfromtimestamp(UTC_ref)}')
        else:
            UTC_timestamp = datetime.datetime.utcfromtimestamp(UTC_ref)
            break
    return UTC_timestamp

def log_attack(attack_msg):
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    response = service.files().get_media(fileId=file_id,supportsAllDrives=True).execute()
    
    new_line = f"\r\n{attack_msg}"
    update_content = response + new_line.encode('utf-8')
    with open(local_file, 'wb') as f: # update log file content
        f.write(update_content)
    
    mime_types = 'text/plain' # mimetype for each file type
    media = MediaFileUpload(local_file, mimetype=mime_types)

    service.files().update(
        fileId=file_id,
        media_body = media,
        supportsAllDrives=True #allow access to share drive
    ).execute()

# Load audio reocrdings
print('Loading audio recordings . . .')
audio_rec, fs = librosa.load(audio_filepath, sr=fs)  # loading the ENF wave
power_rec, fs = librosa.load(power_filepath, sr=fs) 

# Estimate ENF
print('Estimating ENF . . .')
audio_enf = estimate_ENF(audio_rec, 2)
power_enf = estimate_ENF(power_rec,1)

# Correlation Coefficient Analysis
window_size = 120
shift_size = 5

print('Performing correlation coefficient . . .')
rho,total_windows,flag = correlation_vector(audio_enf, power_enf,window_size,shift_size)
t = np.arange(0,total_windows-1,1)

# Log the attack to database if detected
if flag == 1:
    timestamp = UTC_check()
    attack_msg = f'Forgery attack detected | Type: Audio | Timestamp: UTC {timestamp} | Device: {str(dev)}'
    log_attack(attack_msg)

# Plot ENF results
plt.figure(1)
plt.plot(audio_enf,'g', label="Audio Recording")
plt.plot(power_enf,'c', label="Ground Truth")
plt.ylabel('Frequency (Hz)', fontsize=14)
plt.xlabel('Time (sec)', fontsize=14)
plt.title('ENF Fluctuations', fontsize=14)
plt.legend(loc="lower right")
plt.show()

# Display the correlation
plt.figure(2)
plt.plot(t,rho[0][1:],'g--')
plt.hlines(y=0.8, xmin=0, xmax=len(t), colors='r', linestyles='--', lw=2)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xlabel('Number of Windows', fontsize=12)
plt.title('ENF fluctuations compared', fontsize=12)
plt.show()