from src.models.DECOMP import compressor, decompressor
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import shutil
import librosa
from pydub import AudioSegment

# def get_total_duration(folder):
#     total_duration = 0  # Initialize total duration
#
#     # Loop through all files in the folder
#     for file_name in os.listdir(folder):
#         if file_name.endswith(".wav"):
#             file_path = os.path.join(folder, file_name)
#             print(f"Processing {file_path}...")
#
#             # Load the audio file and get its duration
#             audio = AudioSegment.from_file(file_path, format="wav")
#             total_duration += len(audio) / 1000  # Convert from milliseconds to seconds
#
#     return total_duration
#
# # Folder containing WAV files
# folder_path = "/Users/sunhaoran/Database/LibriSpeech/train-clean-100-wav"
#
# # Calculate total duration
# total_duration_seconds = get_total_duration(folder_path)
#
# # Print total duration
# print(f"Total duration of all WAV files: {total_duration_seconds / 3600:.2f} hours")


# def process_audio_folders(input_folder, output_folder, max_duration_seconds=36000):
#     # Ensure the output folder exists
#     os.makedirs(output_folder, exist_ok=True)
#
#     # Get all `b` folders inside the `a` folder
#     b_folders = [os.path.join(input_folder, b) for b in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, b))]
#
#     # Counter for naming the output WAV files
#     file_counter = 1
#     total_duration = 0  # Total duration in seconds
#
#     # Process each `b` folder
#     for b_folder in b_folders:
#         # Get all `c` folders inside the current `b` folder
#         c_folders = [os.path.join(b_folder, c) for c in os.listdir(b_folder) if os.path.isdir(os.path.join(b_folder, c))]
#
#         # Randomly select one `c` folder
#         if not c_folders:
#             print(f"No subfolders in {b_folder}, skipping.")
#             continue
#
#         selected_c_folder = random.choice(c_folders)
#         print(f"Selected folder: {selected_c_folder}")
#
#         # Initialize an empty audio segment
#         combined_audio = AudioSegment.silent(duration=0)
#
#         # Read and combine all FLAC files in the selected `c` folder
#         for file_name in sorted(os.listdir(selected_c_folder)):
#             if file_name.endswith(".flac"):
#                 file_path = os.path.join(selected_c_folder, file_name)
#                 print(f"Processing file: {file_path}")
#                 audio = AudioSegment.from_file(file_path, format="flac")
#                 combined_audio += audio
#
#         # Check the duration of the combined audio
#         combined_duration = len(combined_audio) / 1000  # Duration in seconds
#         if total_duration + combined_duration > max_duration_seconds:
#             print("Reached the maximum duration limit. Stopping.")
#             break

#         # Save the combined audio as a WAV file in the output folder
#         output_file_name = f"{file_counter}.wav"
#         output_file_path = os.path.join(output_folder, output_file_name)
#         combined_audio.export(output_file_path, format="wav")
#         print(f"Saved combined audio to: {output_file_path}")
#
#         # Update counters
#         total_duration += combined_duration
#         file_counter += 1
#
#     print(f"Total duration of generated files: {total_duration / 3600:.2f} hours")
#
# # Input folder containing `b` folders
# input_folder = "/Users/sunhaoran/Database/LibriSpeech/train-clean-100"
#
# # Output folder to save the WAV files
# output_folder = "/Users/sunhaoran/Database/LibriSpeech/train-clean-100-wav"
#
# # Run the function
# process_audio_folders(input_folder, output_folder)

### Create random DRC profiles
# Define the number of items
# num_items = 50
# compressors = {}
# # Generate and populate the dictionary
# for i in range(1, num_items + 1):
#     if i < 26:  # RMS
#         compressors[str(i)] = {
#             'param1': round(random.uniform(-40, 20), 1),
#             'param2': round(random.uniform(1, 10), 1),
#             'param3': round(random.uniform(25, 500), 1),
#             'param4': round(random.uniform(25, 2000), 1),
#             'param5': round(random.uniform(25, 500), 1),
#             'param6': round(random.uniform(25, 2000), 1),
#             'param7': 2  # Constant value
#         }
#     else:  # peak
#         compressors[str(i)] = {
#             'param1': round(random.uniform(-40, 20), 1),
#             'param2': round(random.uniform(1, 10), 1),
#             'param3': round(random.uniform(25, 500), 1),
#             'param4': round(random.uniform(25, 2000), 1),
#             'param5': round(random.uniform(25, 500), 1),
#             'param6': round(random.uniform(25, 2000), 1),
#             'param7': 1  # Constant value
#         }
#
# # save profiles into a pkl file
# f_save = open('50_general_used_drc_5s.pkl', 'wb')
# pickle.dump(compressors, f_save)
# f_save.close()


### 查看wav文件数量
# def count_wav_files(folder_path):
#     # Initialize count
#     count = 0
#     # Iterate over all files in the folder
#     for file_name in os.listdir(folder_path):
#         # Check if the file is a WAV file
#         if file_name.endswith('_0.wav'):
#             count += 1
#     return count
#
# folder_path = '/home/hsun/Datasets/MedleyDB/50profiles_train'
# num_wav_files = count_wav_files(folder_path)
# print(f'Number of WAV files in {folder_path}: {num_wav_files}')


### Create compressed signal dataset
input_folder = "/media/hsun/hsun_exFAT/Datasets/MedleyDB/raw_musics_30"
#
# with open('30profiles_RMS.pkl', 'rb') as file:
#     compressors = pickle.load(file)

compressors = {
    'A': {'param1': -32, 'param2': 3, 'param3': 5, 'param4': 5, 'param5': 13, 'param6': 435, 'param7': 2},
    'B': {'param1': -19.9, 'param2': 1.8, 'param3': 5, 'param4': 5, 'param5': 11, 'param6': 49, 'param7': 2},
    'C': {'param1': -24.4, 'param2': 3.2, 'param3': 5, 'param4': 5, 'param5': 5.8, 'param6': 112, 'param7': 2},
    'D': {'param1': -28.3, 'param2': 7.3, 'param3': 5, 'param4': 5, 'param5': 9, 'param6': 705, 'param7': 2},
    'E': {'param1': -38, 'param2': 4.9, 'param3': 5, 'param4': 5, 'param5': 3.1, 'param6': 257, 'param7': 2},
}

# Make the dataset
duration = 5
# Process WAV Files and Compress
Count = 0
total_num = 0
for file_name in os.listdir(input_folder):
    if file_name.endswith('.wav'):
        # print(file_name)
        Count += 1
        file_path = os.path.join(input_folder, file_name)
        # Load the audio
        audio_data, fs = sf.read(file_path)
        info = sf.info(file_path)
        # Convert to mono
        if info.channels != 1:
            audio_data = np.mean(audio_data, axis=1)
        # audio_data = librosa.resample(audio_data, orig_sr=32000, target_sr=44100)
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
            sf.write('/media/hsun/hsun_exFAT/Datasets/MedleyDB/5profiles/clean/all/' + str(Count) + '_' + str(count) + '_0' + '.wav', segment, fs)
            total_num+=1
            for label, parameters in compressors.items():
                print(Count, count, label)
                y = compressor(segment, fs, parameters.get('param1'), parameters.get('param2'), parameters.get('param3'),
                               parameters.get('param4'), parameters.get('param5'), parameters.get('param6'), parameters.get('param7'))
                sf.write('/media/hsun/hsun_exFAT/Datasets/MedleyDB/5profiles/clean/all/' + str(Count) + '_' + str(count) + '_'
                         + label + '.wav', y, fs)
                total_num+=1
    # if total_num > 35867:
    #     break


# 复制文件
source_folder = "/media/hsun/hsun_exFAT/Datasets/MedleyDB/5profiles/clean/all"
train_folder = "/media/hsun/hsun_exFAT/Datasets/MedleyDB/5profiles/clean/train"
test_folder = "/media/hsun/hsun_exFAT/Datasets/MedleyDB/5profiles/clean/test"

wav_files = [f for f in os.listdir(source_folder) if f.endswith(".wav")]

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
    split_index = int(len(files) * 0.8)  # 80% for train, 20% for test

    train_files = files[:split_index]
    test_files = files[split_index:]

    # Move train files
    for file in train_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))

    # Move test files
    for file in test_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))

print("Files have been successfully split into train and test folders.")

### 查看class数量和对应的文件数量
def count_files_in_classes(folder):
    # Dictionary to store the count of files by class
    class_count = {}

    # Get all files in the folder
    wav_files = [f for f in os.listdir(folder) if f.endswith('.wav')]

    # Group and count files by class
    for wav_file in wav_files:
        class_name = wav_file.split('_')[-1].replace('.wav', '')
        if class_name not in class_count:
            class_count[class_name] = 0
        class_count[class_name] += 1

    return class_count


def print_class_distribution(class_count, folder_name):
    print(f"Class distribution in {folder_name}:")
    for class_name, count in sorted(class_count.items()):
        print(f"Class {class_name}: {count} files")
    print(f"Total classes: {len(class_count)}")
    print(f"Total files: {sum(class_count.values())}\n")


# Paths to train and test folders
train_folder = "/media/hsun/hsun_exFAT/Datasets/MedleyDB/5profiles/clean/train"
test_folder = "//media/hsun/hsun_exFAT/Datasets/MedleyDB/5profiles/clean/test"

# Count files in each class for train and test folders
train_class_count = count_files_in_classes(train_folder)
test_class_count = count_files_in_classes(test_folder)

# Print the distribution
print_class_distribution(train_class_count, "train folder")
print_class_distribution(test_class_count, "test folder")


### 添加噪声
# def add_noise(x, snr):
#     Ps = np.sum(x ** 2)
#     # Signal power, in dB
#     Psdb = 10 * np.log10(Ps)
#     # Noise level necessary
#     Pn = Psdb - snr
#     # Noise vector (or matrix)
#     n = np.sqrt(10 ** (Pn / 10)) * np.random.normal(0, 1, x.shape)
#     return x + n
#
# source_folder = "/opt/data/ljx/shr/DAFX/50profile_test"
# noisy_folder = "/opt/data/ljx/shr/DAFX/50profile_test_noise_-10dB"
#
# print(-10)
#
# for file_name in os.listdir(source_folder):
#     source_file_path = os.path.join(source_folder, file_name)
#     noisy_file_path = os.path.join(noisy_folder, file_name)
#     # Read WAV file
#     audio, fs = sf.read(source_file_path)
#     # Add Gaussian noise with SNR of 20 dB
#     noisy_signal = add_noise(audio, -10)
#     # Save noisy signal as WAV file
#     sf.write(noisy_file_path, noisy_signal, fs)

### Resampling
# def resample_audio(input_filepath, output_filepath, original_sr, target_sr):
#     # Load audio file with original sample rate
#     y, sr = librosa.load(input_filepath, sr=original_sr)
#
#     # Resample the audio
#     y_resampled = librosa.resample(y=y, orig_sr=original_sr, target_sr=target_sr)
#
#     # Save the audio file with new sample rate
#     sf.write(output_filepath, y_resampled, target_sr)
#
# input_folder = '/opt/data/ljx/shr/MedlyDB/raw_musics'
# output_folder = '/opt/data/ljx/shr/MedlyDB/raw_musics_22050'
#
# # Call the function
# for file_name in os.listdir(input_folder):
#     if file_name.endswith('.wav'):
#         file_path = os.path.join(input_folder, file_name)
#         output_path = os.path.join(output_folder, file_name)
#         resample_audio(file_path, output_path, 44100, 22050)
