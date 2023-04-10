# Author: Ruilin Zhang
# Description: Authenticate video recording with ground truth and flag attack to database

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

# Constants for file location
local_file = "updated_log.txt"
folder = "Video_Recording/"
videofile = "short_attack_video.mp4"
powerfile = "power_recording.wav"

video_filepath = folder + videofile
power_filepath = folder + powerfile

# Constants for Rolling Shutter
open_video_to_extract_Row_signal = False # set it to True to extract, else False to use the dump file
pickle_file = videofile[:-4] + "_row_signal.pkl"
video_wave = folder + "mediator.wav"

# ENF function parameters
fs = 1000
nfft = 8192
frame_size = 6
overlap = 0

#funtion to estimate ENF
def estimate_ENF(enf_signal, nominal, harmonics,pyenf_dur):
    enf_signal_object = pyenf.pyENF(signal0=enf_signal, fs=fs, nominal=nominal, harmonic_multiples=harmonics, duration=pyenf_dur,
                                    strip_index=0, frame_size_secs=frame_size, nfft=nfft, overlap_amount_secs=overlap)
    enf_spectro_strip, enf_frequency_support = enf_signal_object.compute_spectrogam_strips()
    enf_weights = enf_signal_object.compute_combining_weights_from_harmonics()
    enf_OurStripCell, enf_initial_frequency = enf_signal_object.compute_combined_spectrum(enf_spectro_strip,
                                                                                          enf_weights,
                                                                                          enf_frequency_support)
    ENF = enf_signal_object.compute_ENF_from_combined_strip(enf_OurStripCell, enf_initial_frequency)
    return ENF[:-10]

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
        else:
            cnt = 0

        if (cnt >= attack_threshold):
            attack_flag = 1
    
    return rho,total_windows, attack_flag

def extract_row_pixel(frame):
    # check for grayscale or RGB
    frame_shape = frame.shape
    if frame_shape[2] == 3:  # its an RGB frame
        average_frame_across_rgb = np.mean(frame, axis=2)
        average_frame_across_column = np.mean(average_frame_across_rgb, axis=1)
    else:
        average_frame_across_column = np.mean(frame, axis=1)
    average_frame_across_column = np.reshape(average_frame_across_column, (frame_shape[0],))
    return average_frame_across_column

def rolling_shutter_extraction():
    # Input the video stream
    video = cv2.VideoCapture(video_filepath)

    # Validating the read of input video
    if not video.isOpened():
        print("Error Opening the video stream or file")

    # Video specifics extraction
    total_number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    height_of_frame = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # total number of rows
    width_of_frame = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # total number of columns
    frame_rate = float(video.get(cv2.CAP_PROP_FPS))
    size_of_row_signal = int(np.multiply(total_number_of_frames, height_of_frame))
    row_signal = np.zeros((total_number_of_frames, height_of_frame, 1), dtype=float)

    if open_video_to_extract_Row_signal is True:
        frame_counter = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret is True:
                row_signal[frame_counter, :, 0] = extract_row_pixel(frame)
                cv2.imshow('Frame', frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
            frame_counter += 1
        video.release()
        cv2.destroyAllWindows()
        # store the variables for faster future use
        variable_location = folder + pickle_file
        store_variable_file = open(variable_location, 'wb')
        pickle.dump(row_signal, store_variable_file)
        store_variable_file.close()
        print("Extracted Row Signal and stored in dump.\n")
    else:
        variable_location = folder + pickle_file
        load_variable_file = open(variable_location, 'rb')
        row_signal = pickle.load(load_variable_file)
        load_variable_file.close()
        print("Loaded the Row Signal. \n")
    
    # For a static video, clean the row signal with its video signal
    # that should leave only the ENF signal
    # Refer to { Exploiting Rolling Shutter For ENF Signal Extraction From Video }
    # row_signal = video_signal + enf_signal
    # average_of_each_row_element(row_signal) = average_of_each_row_element(video_signal) [since average of enf is 0]
    # enf_signal = row_signal - average_of_each_row_element(row_signal)

    # Estimate the ENF signal using the row signal collected
    average_of_each_row_element = np.mean(row_signal, axis=0)
    enf_video_signal = row_signal - average_of_each_row_element
    flattened_enf_signal = enf_video_signal.flatten()  # the matrix shape ENF data is flattened to one dim data
    # Writing the ENF data to the wav file for data type conversion
    scipy.io.wavfile.write(video_wave, rate=int(frame_rate * height_of_frame), data=flattened_enf_signal)

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

# Performing Rolling Shutter Extraction
rolling_shutter_extraction()

# Load video reocrdings
print('Loading video recordings . . .')
video_rec, fs = librosa.load(video_wave, sr=fs)  # loading the ENF wave
power_rec, fs = librosa.load(power_filepath, sr=fs) 

# Estimate ENF
print('Estimating ENF . . .')
video_nominal = 120
video_dur = 1
power_nominal = 60
power_dur = 0.1
video_enf = estimate_ENF(video_rec, video_nominal, 2, video_dur)
power_enf = estimate_ENF(power_rec, power_nominal, 1, power_dur)

# Correlation Coefficient Analysis
window_size = 50
shift_size = 1

print('Performing correlation coefficient . . .')
rho,total_windows,flag = correlation_vector(video_enf, power_enf,window_size,shift_size)
t = np.arange(0,total_windows-1,1)

# Log the attack to database if detected
if flag == 1:
    timestamp = UTC_check()
    attack_msg = f'Forgery attack detected | Type: Video | Timestamp: UTC {timestamp} | Device: Laptop'
    log_attack(attack_msg)

# Plot ENF results
fig, (video, power) = plt.subplots(2, 1, sharex=True)
video.plot(video_enf,'r', label="Video Recording")
video.set_title("Video ENF Signal", fontsize=12)
video.ticklabel_format(useOffset=False)
video.set_ylabel("Freq (Hz)", fontsize=12)

power.plot(power_enf,'g', label="Ground Truth")
power.set_title("Power ENF Signal", fontsize=12)
power.set_ylabel("Freq (Hz)", fontsize=12)
power.set_xlabel("Time", fontsize=12)
plt.show()

# Display the correlation
plt.figure(2)
plt.plot(t,rho[0][1:],'g--')
plt.hlines(y=0.8, xmin=0, xmax=len(t), colors='r', linestyles='--', lw=2)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xlabel('Number of Windows', fontsize=12)
plt.title('ENF fluctuations compared', fontsize=12)
plt.show()