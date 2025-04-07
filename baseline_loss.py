import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from audio_loss import loss_mse, MelSTFTLoss, loss_SISDR
import pandas as pd

ref_folder = "/home/hsun/Datasets/LibriSpeech/5profiles_noised_small"
restored_folder = "/home/hsun/Datasets/LibriSpeech/5profiles_test_reg_noised_small"
# restored_folder = "/home/hsun/Datasets/LibriSpeech/30profiles_RMS_test_class"
# restored_folder = "/home/hsun/Projects/de-limiter/De-limiter/output_musdb"

Mel = MelSTFTLoss()

# List to store MSE values
mse_values = []
mel_values = []
sisdr_values = []

wav_files_b = sorted([f for f in os.listdir(restored_folder) if f.endswith('.wav')])

# Calculate MSE for each pair of WAV files
for filename in wav_files_b:
    # Read WAV file
    data_b, rate_b = sf.read(os.path.join(restored_folder, filename))

    info = sf.info(os.path.join(restored_folder, filename))
    if info.channels != 1:
        data_b = np.mean(data_b, axis=1)

    name_parts = filename.split('_')
    name_parts[-1] = '0.wav'
    new_filename = '_'.join(name_parts)

    data_a, rate_a = sf.read(os.path.join(ref_folder, new_filename))

    if rate_b !=  rate_a:
        data_b = librosa.resample(data_b, orig_sr=rate_b, target_sr=rate_a)

    # Calculate MSE
    mse = loss_mse(data_b, data_a)
    mse_values.append(mse)

    mel = Mel.compute_loss(data_b, data_a)
    mel_values.append(mel)

    sisdr = loss_SISDR(data_b, data_a)
    sisdr_values.append(sisdr)

# Calculate mean and standard deviation of MSE
mean_mse = np.mean(mse_values)
std_mse = np.std(mse_values)

mean_mel = np.mean(mel_values)
std_mel = np.std(mel_values)

mean_sisdr = np.mean(sisdr_values)
std_sisdr = np.std(sisdr_values)

# plt.figure()
# box_data = [mse_values, mel_values, sisdr_values]
# plt.boxplot(box_data,
#             labels=['MSE', 'Mel', 'SISDR'],
#             showfliers=False)
# plt.ylabel('Loss Value')
# plt.show(block=True)

print(f'MSE Loss: {mean_mse}, Mel Loss: {mean_mel}, SISDR: {mean_sisdr}, \n std MSE: {std_mse}, std Mel: {std_mel}, std SISDR: {std_sisdr}')

# loss_tab = {
#     "MSE": [],
#     "Mel": [],
#     "SI-SDR": []
# }
#
# loss_tab["MSE"].append(np.array(mse_values).T)
# loss_tab["Mel"].append(np.array(mel_values).T)
# loss_tab["SI-SDR"].append(np.array(sisdr_values).T)

box_data = [mse_values, mel_values, sisdr_values]
df = pd.DataFrame(np.array(box_data).T)
df.to_excel("../results/loss_LibriSpeech_5_noised_small.xlsx", index=False)