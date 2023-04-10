# Author: Ruilin Zhang
# Description: Display recent files in Google Drive database

import csv
from datetime import datetime
from googleapiclient.http import MediaFileUpload
from Google import Create_Service
import io
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

## Create google drive service instance
CLIENT_SECRET_FILE = 'credentials.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']

# Database Folder ID
dev1 = '1R1TKGnGrzPTUfJHYA8eKvTeK_XabApw4'
dev2 = '1wnN0MxzPHN2iBy35TgZYS5VB1dzzKMdo'
dev3 = '1YfZilhQKiLEwElLEbxzuaJLmVbxHje3v'
dev4 = '14M1cek8f7k5Y90kZLR8fCGJcFoyfPJqu'
dev5 = '1bTsof1ewrz_BmH6gpq34PzJLMIkA5hBa'
dev6 = '1m_1g7cv4_LnU6Du1B7RkS1p9-96ra4qb'

devs = [dev1, dev2, dev3, dev4, dev5, dev6]
ref_node = dev5 # device always operating
csv_data_tot = 1800 # 1 hour, frame_size = 2

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

def list_files(query):
    # String to specify what to query
    # Must be correct parent ID, name requirement, exclude trashed data
    response = service.files().list(q=query,
                                    orderBy='createdTime desc', 
                                    includeItemsFromAllDrives=True,
                                    supportsAllDrives=True
                                    ).execute()

    files = response.get('files') # object of file metadata
    df = pd.DataFrame(files)

    return df

def scan_folder(id):
    mime_types = 'text/csv'
    query = f"parents = '{id}' and name contains 'Dev' and trashed = false and mimeType = '{mime_types}'"

    df = list_files(query)

    if len(df) > 1: # number of files
        file_id = df.get('id')[1]
        filename = df.get('name')[1]
    else:
        file_id = df.get('id')[0]
        filename = df.get('name')[0] # most recent
    return file_id, filename

def load_file(target_date, target_csv):
    ENF_matrix = np.empty((0,csv_data_tot))
    for device_id in devs: # go through each device folder URL ID
        query = f"parents = '{device_id}'and name contains '{target_date}' and trashed = false" # finds file with correct date
        df = list_files(query)

        if len(df) > 0:
            folder_date_id = df.get('id')[0]
            mime_types = 'text/csv'
            query = f"parents = '{folder_date_id}'and name contains '{target_csv}' and trashed = false and mimeType = '{mime_types}' " # finds csv file
            df = list_files(query)
            if len(df) > 0:
                folder_date_id = df.get('id')[0]
                ENF = read_csv(folder_date_id)
            else:
                ENF = np.full(csv_data_tot,-1)
        else:
            ENF = np.full(csv_data_tot,-1) # -1 indicate no ENF data present

        ENF_matrix = np.vstack((ENF_matrix, ENF)) # append ENF in as rows
    
    return ENF_matrix

def read_csv(file_id):
    request = service.files().get_media(fileId=file_id)
    content = request.execute()
    content_string = content.decode('utf-8') # convert byte string to regular string

    csv_reader = csv.reader(io.StringIO(content_string)) # convert string to file-like object for reading
    next(csv_reader) #skip 1st line

    data = []
    for item in csv_reader:
        data = np.append(data, float(item[1]))
    
    return data

def restart():
    print("argv: ", sys.argv)
    print("sys executable: ", sys.executable)
    print("Restart now!")
    os.execv(sys.executable, ['python'] + sys.argv)

def main():
    # Check most recent date in database
    # Choose device that should always be operating
    
    query = f"parents = '{ref_node}' and name contains 'Dev' and trashed = false"
    df = list_files(query)
    recent_id = df.get('id')[0]
    recent_name = df.get('name')[0] # item listed by descending creation time (first item most recent)

    file_id, filename = scan_folder(recent_id)
    target_date = recent_name[5:]
    target_csv = filename[5:-4]
    """
    # Manually select the date and hour, cannot plot very old files (4+ days)
    target_date = 'Power_ENF_2023_03_27'
    target_csv = 'ENF_Hr00'
    """
    ENF = load_file(target_date, target_csv) # load ENF matrix from all devices
    timestamp = target_date[10:] + ' ' + target_csv[4:]
    print(f'UTC Timestamp: {timestamp}')
    color_table = ['mediumpurple', 'crimson', 'gold', 'lightpink', 'palegreen', 'deepskyblue']

    buffer_chunk = 1
    chunk_size = 60 - buffer_chunk
    y_center = 60 # nominal 60 hz
    y_offset = 0.03

    for i in range(0, len(ENF[0]), buffer_chunk):
        data_x = np.arange(i,i+chunk_size)
        for j in range(0,6):
            if(ENF[j][0] != -1):
                plt.plot(data_x, ENF[j][i:i+chunk_size], color_table[j], label=f'Dev{j+1}')
        plt.ylabel('Frequency (Hz)', fontsize=14)
        plt.xlabel('ENF Data (1 Data: 2 sec.)', fontsize=14)
        #plt.ylim(y_center - y_offset, y_center + y_offset)
        plt.title(f'ENF Fluctuations for {timestamp}', fontsize=14)
        plt.legend(loc="lower right")
        
        plt.pause(0.1) # refresh every 1 sec.

        plt.clf()
    
    restart()

if __name__ == '__main__':
    main()