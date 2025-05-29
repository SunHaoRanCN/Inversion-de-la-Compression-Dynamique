from models.DECOMP import compressor
import os
import soundfile as sf
import numpy as np
import random
import pickle
import shutil


### Create compressed signal dataset
input_folder = " "
target_folder = " "
all_folder = os.path.join(target_folder, "all")
train_folder = os.path.join(target_folder, "train")
test_folder = os.path.join(target_folder, "test")
os.makedirs(all_folder, exist_ok=True)
os.makedirs(all_folder, exist_ok=True)
os.makedirs(all_folder, exist_ok=True)

compressors = pickle.load(open(" ", 'rb'))

# Make the dataset
duration = 5
# Process WAV Files and Compress
Count = 0
total_num = 0

for file_name in os.listdir(input_folder):
    if file_name.endswith('.wav'):
        Count += 1
        file_path = os.path.join(input_folder, file_name)
        audio_data, fs = sf.read(file_path)
        info = sf.info(file_path)

        # Convert to mono
        if info.channels != 1:
            audio_data = np.mean(audio_data, axis=1)

        # Calculate the number of 5-second segments
        num_segments = int(len(audio_data) // (fs * duration))

        # Loop through each segment and save it as a new WAV file
        count = 0
        for i in range(num_segments):
            start_time = int(i * fs * duration)
            end_time = int((i + 1) * fs * duration)
            segment = audio_data[start_time:end_time]

            if np.all(segment == 0):
                continue

            E = 10 * np.log10(np.sum(segment ** 2))
            if E < -30:
                continue

            # normalization
            segment = segment - np.mean(segment)
            segment = segment / np.max(np.abs(segment))
            count += 1

            sf.write(os.path.join(all_folder, str(Count), '_', str(count), '_0', '.wav'), segment, fs)

            for label, parameters in compressors.items():
                y = compressor(segment, fs, parameters.get('param1'), parameters.get('param2'), parameters.get('param3'),
                               parameters.get('param4'), parameters.get('param5'), parameters.get('param6'), parameters.get('param7'))
                sf.write(os.path.join(all_folder, str(Count), '_', str(count), '_', label, '.wav'), y, fs)


### Train-test-split
split_ratio = 0.8

wav_files = [f for f in os.listdir(all_folder) if f.endswith(".wav")]

# Dictionary to store files by class
files_by_class = {}

# Group files by class
for wav_file in wav_files:
    class_name = wav_file.split('_')[-1].replace('.wav', '')
    if class_name not in files_by_class:
        files_by_class[class_name] = []
    files_by_class[class_name].append(wav_file)

# Split files into train and test sets
for class_name, files in files_by_class.items():
    random.shuffle(files)
    split_index = int(len(files) * split_ratio)

    train_files = files[:split_index]
    test_files = files[split_index:]

    # Move train files
    for file in train_files:
        shutil.copy(os.path.join(all_folder, file), os.path.join(train_folder, file))

    # Move test files
    for file in test_files:
        shutil.copy(os.path.join(all_folder, file), os.path.join(test_folder, file))

print("Files have been successfully split into train and test folders.")
