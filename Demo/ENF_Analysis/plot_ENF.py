import csv
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import math
from scipy.stats.stats import pearsonr

# Constants for file location

folder1 = 'Dev1/'
folder2 = 'Dev2/'
folder3 = 'Dev3/'
folder4 = 'Dev4/'
folder5 = 'Dev5/'
folder6 = 'Dev6/'

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

def read_enf_signal(filepath):
    with open(filepath, 'r') as csv_file:
        ENF = []
        csv_reader = csv.reader(csv_file)

        next(csv_reader) #skip 1st line

        for item in csv_reader:
            ENF = np.append(ENF, float(item[1]))
    
    return ENF

def main():
    ENF1 = []
    
    list_files = os.listdir(folder1)

    for filename in list_files: # compute ENF in every file
        filepath = folder1 + filename

        read_ENF = read_enf_signal(filepath)
        ENF1 = np.append(ENF1, read_ENF)
    
    ENF2 = []
    
    list_files = os.listdir(folder2)

    for filename in list_files: # compute ENF in every file
        filepath = folder2 + filename

        read_ENF = read_enf_signal(filepath)
        ENF2 = np.append(ENF2, read_ENF)
    
    ENF3 = []
    
    list_files = os.listdir(folder3)

    for filename in list_files: # compute ENF in every file
        filepath = folder3 + filename

        read_ENF = read_enf_signal(filepath)
        ENF3 = np.append(ENF3, read_ENF)

    ENF4 = []
    
    list_files = os.listdir(folder4)

    for filename in list_files: # compute ENF in every file
        filepath = folder4 + filename

        read_ENF = read_enf_signal(filepath)
        ENF4 = np.append(ENF4, read_ENF)
    
    ENF5 = []
    
    list_files = os.listdir(folder5)

    for filename in list_files: # compute ENF in every file
        filepath = folder5 + filename

        read_ENF = read_enf_signal(filepath)
        ENF5 = np.append(ENF5, read_ENF)

    ENF6 = []
    
    list_files = os.listdir(folder6)

    for filename in list_files: # compute ENF in every file
        filepath = folder6 + filename

        read_ENF = read_enf_signal(filepath)
        ENF6 = np.append(ENF6, read_ENF)

    ref_node = ENF1
    test_node = ENF2

    fig, (ENF, Correlation) = plt.subplots(2, 1)
    ENF.plot(ref_node,'g', label="Dev1")
    ENF.plot(test_node,'c', label="Dev2")
    ENF.set_ylabel('Frequency (Hz)', fontsize=14)
    ENF.set_xlabel('Time (sec)', fontsize=14)
    ENF.set_title('24-Hour ENF Fluctuations', fontsize=14)
    ENF.legend(loc="lower right")
    
    window_size = 60
    shift_size = 5
    rho,total_windows = correlation_vector(ref_node, test_node,window_size,shift_size)
    
    # Display the correlation

    t = np.arange(0,total_windows-1,1)
    Correlation.plot(t,rho[0][1:],'g--', label="Plain Wall")
    Correlation.hlines(y=0.8, xmin=0, xmax=len(t), colors='r', linestyles='--', lw=2)
    Correlation.set_ylabel('Correlation Coefficient', fontsize=12)
    Correlation.set_xlabel('Number of Windows compared', fontsize=12)
    Correlation.set_title('ENF fluctuations compared', fontsize=12)
    Correlation.legend(loc="lower right")
    plt.show()
    
    test_node = ENF3

    fig, (ENF, Correlation) = plt.subplots(2, 1)
    ENF.plot(ref_node,'g', label="Dev1")
    ENF.plot(test_node,'c', label="Dev3")
    ENF.set_ylabel('Frequency (Hz)', fontsize=14)
    ENF.set_xlabel('Time (sec)', fontsize=14)
    ENF.set_title('24-Hour ENF Fluctuations', fontsize=14)
    ENF.legend(loc="lower right")
    
    rho,total_windows = correlation_vector(ref_node, test_node,window_size,shift_size)

    # Display the correlation
    t = np.arange(0,total_windows-1,1)
    Correlation.plot(t,rho[0][1:],'g--', label="Plain Wall")
    Correlation.hlines(y=0.8, xmin=0, xmax=len(t), colors='r', linestyles='--', lw=2)
    Correlation.set_ylabel('Correlation Coefficient', fontsize=12)
    Correlation.set_xlabel('Number of Windows compared', fontsize=12)
    Correlation.set_title('ENF fluctuations compared', fontsize=12)
    Correlation.legend(loc="lower right")
    plt.show()

    test_node = ENF4

    fig, (ENF, Correlation) = plt.subplots(2, 1)
    ENF.plot(ref_node,'g', label="Dev1")
    ENF.plot(test_node,'c', label="Dev4")
    ENF.set_ylabel('Frequency (Hz)', fontsize=14)
    ENF.set_xlabel('Time (sec)', fontsize=14)
    ENF.set_title('24-Hour ENF Fluctuations', fontsize=14)
    ENF.legend(loc="lower right")
    
    rho,total_windows = correlation_vector(ref_node, test_node,window_size,shift_size)

    # Display the correlation
    t = np.arange(0,total_windows-1,1)
    Correlation.plot(t,rho[0][1:],'g--', label="Plain Wall")
    Correlation.hlines(y=0.8, xmin=0, xmax=len(t), colors='r', linestyles='--', lw=2)
    Correlation.set_ylabel('Correlation Coefficient', fontsize=12)
    Correlation.set_xlabel('Number of Windows compared', fontsize=12)
    Correlation.set_title('ENF fluctuations compared', fontsize=12)
    Correlation.legend(loc="lower right")
    plt.show()

    test_node = ENF5

    fig, (ENF, Correlation) = plt.subplots(2, 1)
    ENF.plot(ref_node,'g', label="Dev1")
    ENF.plot(test_node,'c', label="Dev5")
    ENF.set_ylabel('Frequency (Hz)', fontsize=14)
    ENF.set_xlabel('Time (sec)', fontsize=14)
    ENF.set_title('24-Hour ENF Fluctuations', fontsize=14)
    ENF.legend(loc="lower right")
    
    rho,total_windows = correlation_vector(ref_node, test_node,window_size,shift_size)

    # Display the correlation
    t = np.arange(0,total_windows-1,1)
    Correlation.plot(t,rho[0][1:],'g--', label="Plain Wall")
    Correlation.hlines(y=0.8, xmin=0, xmax=len(t), colors='r', linestyles='--', lw=2)
    Correlation.set_ylabel('Correlation Coefficient', fontsize=12)
    Correlation.set_xlabel('Number of Windows compared', fontsize=12)
    Correlation.set_title('ENF fluctuations compared', fontsize=12)
    Correlation.legend(loc="lower right")
    plt.show()

    test_node = ENF6

    fig, (ENF, Correlation) = plt.subplots(2, 1)
    ENF.plot(ref_node,'g', label="Dev1")
    ENF.plot(test_node,'c', label="Dev6")
    ENF.set_ylabel('Frequency (Hz)', fontsize=14)
    ENF.set_xlabel('Time (sec)', fontsize=14)
    ENF.set_title('24-Hour ENF Fluctuations', fontsize=14)
    ENF.legend(loc="lower right")
    
    rho,total_windows = correlation_vector(ref_node, test_node,window_size,shift_size)

    # Display the correlation
    t = np.arange(0,total_windows-1,1)
    Correlation.plot(t,rho[0][1:],'g--', label="Plain Wall")
    Correlation.hlines(y=0.8, xmin=0, xmax=len(t), colors='r', linestyles='--', lw=2)
    Correlation.set_ylabel('Correlation Coefficient', fontsize=12)
    Correlation.set_xlabel('Number of Windows compared', fontsize=12)
    Correlation.set_title('ENF fluctuations compared', fontsize=12)
    Correlation.legend(loc="lower right")
    plt.show()
    
if __name__ == '__main__':
    main()

