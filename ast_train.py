import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import librosa
import soundfile as sf
import torch.backends.cudnn as cudnn
from ast_model import ASTModel
import time
from torchinfo import summary
import pandas as pd
import random

cudnn.benchmark = True

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时
    torch.backends.cudnn.deterministic = True  # 确保卷积算法确定性
    torch.backends.cudnn.benchmark = False  # 关闭自动优化（固定计算流程）

    # NumPy
    np.random.seed(seed)

    # Python random
    random.seed(seed)

    # 设置DataLoader的随机种子（多进程时）
    # def _init_fn(worker_id):
    #     np.random.seed(seed + worker_id)
    #
    # return _init_fn

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, resample=False, target_fs=16000, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.resample = resample
        self.target_fs = target_fs
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        type = self.labels[idx]
        audio, sr = sf.read(audio_path)

        if self.resample == True:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_fs)
            sr = self.target_fs

        total_samples = 5 * sr
        target_shape = (64, 431)
        n_fft = 2 * (target_shape[0] - 1)
        hop_length = (total_samples - n_fft) // (target_shape[1] - 1)
        # mel = librosa.feature.melspectrogram(y=audio, sr=0.5*sr, n_fft=2048, hop_length=128, win_length=2048, n_mels=128)  # 5s audio LibriSpeech
        # mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_fft=2048, hop_length=205, win_length=2048, n_mels=64)  # 2s audio LibriSpeech
        # mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_fft=2048, hop_length=512, win_length=2048, n_mels=128)
        ###stft
        # mel = librosa.stft(y=audio,n_fft=254, hop_length=512) # (128, 431) 44100
        mel = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)  # (64, 431) 44100
        mel = librosa.amplitude_to_db(np.abs(mel))

        mel = mel.T
        # mel = imageProcess(mel, (154*3, 12*3))

        if self.transform:
            mel = self.transform(mel)

        return mel, type

# Hyperparameters
seed = 22   # 11 -> MLP, 22 -> KAN
set_seed(seed)
batch_size = 12
lr = 0.0001
n_epochs = 3
norm_param = 0.001

### Data loading and preprocessing
audio_folder_path = "/home/hsun/Datasets/LibriSpeech/5profiles"
train_folder = "/home/hsun/Datasets/LibriSpeech/5profiles_train"
test_folder = "/home/hsun/Datasets/LibriSpeech/5profiles_test"
# audio_folder_path = "/home/hsun/Datasets/LibriSpeech/30profiles_RMS"
# train_folder = "/home/hsun/Datasets/LibriSpeech/30profiles_RMS_train"
# test_folder = "/home/hsun/Datasets/LibriSpeech/30profiles_RMS_test"


def find_paths_labels(folder_path):
    all_files = []
    all_labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            parts = file_name.split('_')
            # label = int(parts[2].split('.')[0])
            label = parts[2].split('.')[0]
            if label == 'O':
                label = 0
            elif label == 'A':
                label = 1
            elif label == 'B':
                label = 2
            elif label == 'C':
                label = 3
            elif label == 'D':
                label = 4
            else:
                label = 5
            all_files.append(file_path)
            all_labels.append(label)
    return all_files, all_labels

train_files, train_labels = find_paths_labels(train_folder)
test_files, test_labels = find_paths_labels(test_folder)
# audio_files, labels = find_paths_labels(audio_folder_path)

train_dataset = AudioDataset(train_files, train_labels, resample=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

test_dataset = AudioDataset(test_files, test_labels, resample=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize model, loss function, and optimizer
model = ASTModel(label_dim=31, input_tdim=431, input_fdim=64, imagenet_pretrain=True, audioset_pretrain=False)
# print(model)
model = nn.DataParallel(model)
model = model.to(device)
summary(model, estimate_size = (batch_size, 1, 220500), device = device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
expLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

# Training loop with early stopping
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

### Early stopping parameters
min_loss_val = 10
patience = 30  # Number of epochs with no improvement after which training will be stopped
best_val_loss = float('inf')
patience_counter = 0

best_test_acc = 0

early_stop = True
if early_stop:
    print("Early stop with patience {}".format(patience))
else:
    print("Training for {} epochs without early stop".format(n_epochs))

torch.cuda.synchronize()
start = time.time()

for epoch in range(n_epochs):
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    model.train()
    running_loss = 0.0
    correct_predictions_train = 0
    total_samples_train = 0
    all_labels_train = []
    all_predictions_train = []
    for inputs, labels in train_loader:
        inputs = inputs.to(torch.float32)
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # l2_lambda = norm_param
        # l2_regularization = torch.tensor(0).to(device).to(torch.float32)
        # for param in model.parameters():
        #     l2_regularization += torch.norm(param, p=2)
        # loss += l2_lambda * l2_regularization

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_samples_train += labels.size(0)
        correct_predictions_train += (predicted == labels).sum().item()

        all_labels_train.extend(labels.cpu().numpy())
        all_predictions_train.extend(predicted.cpu().numpy())
    expLR.step()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions_train / total_samples_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    val_loss = 0.0
    correct_predictions_val = 0
    total_samples_val = 0
    all_labels = []
    all_predictions = []
    rightFiles = []
    wrongFiles = []
    count = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total_samples_val += labels.size(0)
            correct_predictions_val += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss /= len(test_loader)
    val_accuracy = correct_predictions_val / total_samples_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
          f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

    # if val_loss <= min_loss_val:
    #     min_loss_val = val_loss
    #     best_model = model.state_dict()
    #     torch.save(best_model, 'pretrained_ast_LibriSpeech_stft_431_128.pt')

    ### Early stopping check
    # if early_stop:
    #     if epoch < 30:
    #         continue
    #     else:
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             patience_counter = 0
    #         else:
    #             patience_counter += 1
    #
    #     if patience_counter >= patience:
    #         print(f'Early stopping after {epoch + 1} epochs without improvement.')
    #         break

# torch.save(best_model, 'ast_best_model.pth')

torch.cuda.synchronize()
end = time.time()

training_time_hours = (end - start) / 3600
time_per_epoch = training_time_hours / (epoch + 1)

### Plotting the learning curve
# plt.figure(figsize=(12, 6))
# # Plot Training Loss
# plt.subplot(2, 2, 1)
# plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# # Plot Validation Loss
# plt.subplot(2, 2, 2)
# plt.plot(range(1, len(train_losses) + 1), val_losses, label='Validation Loss', color='orange')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Validation Loss')
# # Plot Training Accuracy
# plt.subplot(2, 2, 3)
# plt.plot(range(1, len(train_losses) + 1), train_accuracies, label='Training Accuracy', color='green')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy')
# # Plot Validation Accuracy
# plt.subplot(2, 2, 4)
# plt.plot(range(1, len(train_losses) + 1), val_accuracies, label='Validation Accuracy', color='red')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Validation Accuracy')
# plt.tight_layout()
# plt.savefig("../results/lcurve_LibriSpeech_626_128.png")
#
# # Calculate confusion matrix for validation set
# conf_matrix = confusion_matrix(all_labels, all_predictions)
# print(conf_matrix)
# plt.figure(figsize=(8, 6))
# plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.colorbar()
# plt.savefig('../results/conf_matice_LibriSpeech_626_128.png')
#
# score_matrix = classification_report(all_labels, all_predictions, output_dict=True, digits=4)
# print(score_matrix)
# df = pd.DataFrame(score_matrix).transpose()
# df.to_csv('../results/score_LibriSpeech_626_128.csv')

best_val_accuracy = max(val_accuracies)
best_epoch = val_accuracies.index(best_val_accuracy) + 1
print(f'Best validation accuracy: {best_val_accuracy:.4f} achieved at epoch {best_epoch}')
print(f'Average training time: {time_per_epoch:.4f} h')


#############################################################
# compressors = {
#     'A': {'param1': -32, 'param2': 3, 'param3': 5, 'param4': 5, 'param5': 13, 'param6': 435, 'param7': 2},
#     'B': {'param1': -19.9, 'param2': 1.8, 'param3': 5, 'param4': 5, 'param5': 11, 'param6': 49, 'param7': 2},
#     'C': {'param1': -24.4, 'param2': 3.2, 'param3': 5, 'param4': 5, 'param5': 5.8, 'param6': 112, 'param7': 2},
#     'D': {'param1': -28.3, 'param2': 7.3, 'param3': 5, 'param4': 5, 'param5': 9, 'param6': 705, 'param7': 2},
#     'E': {'param1': -38, 'param2': 4.9, 'param3': 5, 'param4': 5, 'param5': 3.1, 'param6': 257, 'param7': 2},
# }
# compressors_torch = {key: {param: torch.tensor(value).to(device) for param, value in params.items()} for key, params
#                      in compressors.items()}
#
#
# def prediction_to_label(predictions):
#     p = []
#     for i in predictions:
#         if i == 0:
#             l = 'O'
#         elif i == 1:
#             l = 'A'
#         elif i == 2:
#             l = 'B'
#         elif i == 3:
#             l = 'C'
#         elif i == 4:
#             l = 'D'
#         else:
#             l = 'E'
#         p.append(l)
#     return p
#
# predicted_labels = prediction_to_label(all_predictions)
#
#
# def checkFunction(filepath, Compressor):
#     profile_label = filepath[-5]
#     audio, fs = sf.read(filepath)
#     if profile_label != 'O':
#         parameters = Compressor[profile_label]
#         original_file = filepath[:-5] + 'O' + filepath[-4:]
#         oriAudio, _ = sf.read(original_file)
#         y, my_g, v, ff, xx = compressor(oriAudio, fs, parameters.get('param1'), parameters.get('param2'),
#                                      parameters.get('param3'), parameters.get('param4'), parameters.get('param5'),
#                                      parameters.get('param6'), parameters.get('param7'))
#         f1 = np.square(my_g-1)
#         f1sum = np.sum(f1)
#         # f2 = np.square(oriAudio - y)
#         # f2sum = np.sum(f2)
#         # f3 = np.std(my_g)
#     else:
#         # my_g = np.ones(len(audio))
#         # f1 = f2 = np.zeros(len(audio))
#         f1sum = f2sum = 0
#         # f3 = 1
#     return f1sum
#
#
# rightFiles = np.concatenate(rightFiles)
# wrongFiles = np.concatenate(wrongFiles)
# combined_list = [(path, 'y') for path in rightFiles] + [(path, 'n') for path in wrongFiles]
# groups = defaultdict(lambda: defaultdict(list))
#
# for path, classification in combined_list:
#     _, file_name = path.rsplit('/', 1)  # Extract the file name
#     song_index, seg_index, profile = file_name.split('_')[:3]
#     check_value = checkFunction(path, compressors)
#     groups[profile][song_index].append((seg_index, path, classification, check_value))
#
# for profile, songs in groups.items():
#     for song_index, paths in songs.items():
#         sorted_paths = natsorted(paths, key=lambda x: x[0])
#         groups[profile][song_index] = [(path, classification, check_value)
#                                        for _, path, classification, check_value in sorted_paths]
#
# num_profiles = len(groups)
# fig, axs = plt.subplots(num_profiles, 1, sharex=True, figsize=(16, 16))
# # Iterate over profiles and draw point diagrams for each songindex
# for i, (profile, songs) in enumerate(groups.items()):
#     axs[i].set_title(f"Profile: {profile}")
#     for song_index, segments in songs.items():
#         segments = np.array(segments)
#         paths, classifications, check_values = np.transpose(segments)
#         # Set color based on classification
#         colors = ['red' if i == 'n' else 'blue' for i in classifications]
#         axs[i].scatter(np.arange(len(check_values)), check_values.astype(float), color=colors)
#     axs[i].set_xlabel('Seg Index')
#     axs[i].set_ylabel(r'$sum(|g-1|^2)$')
# plt.tight_layout()
# path_1 = '/home/ljx/0130/'
# name_1 = 'ast_checkFunction_4.png'
# plt.savefig(path_1 + name_1)
#
#
# def loss_mel(x, hat_x, fs=22050, n_fft=2048, hop_length=512, win_length=2048, window='hann', n_mels=128):
#     mel_spec1 = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
#                                                n_mels=n_mels, window=window)
#     log_mel_spec1 = librosa.power_to_db(mel_spec1, ref=np.max)
#     mel_spec2 = librosa.feature.melspectrogram(y=hat_x, sr=fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
#                                                n_mels=n_mels, window=window)
#     log_mel_spec2 = librosa.power_to_db(mel_spec2, ref=np.max)
#     l1_norm = np.sum(np.abs(log_mel_spec1 - log_mel_spec2))
#     return l1_norm
#
# MSE_loss = []
# Mel_loss = []
# for file_path in X_test:
#     audio, sr = sf.read(file_path)
#     Index = X_test.index(file_path)
#     predicted_label = predicted_labels[Index]
#     # compression inversion
#     if predicted_label == 'O':
#         x_hat_tensor = torch.tensor(audio).to(device)
#     else:
#         parameters = compressors_torch[predicted_label]
#         x_hat_tensor, g, v, x2 = decompressor_torch(torch.tensor(audio).to(device), sr, parameters.get('param1'), parameters.get('param2'),
#                                        parameters.get('param3'), parameters.get('param4'), parameters.get('param5'),
#                                        parameters.get('param6'), parameters.get('param7'))
#     # find the real signal
#     original_audio_path = file_path[:-5] + 'O' + '.wav'
#     original_audio, _ = sf.read(original_audio_path)
#     # compute errors
#     loss_MSE = rmse_torch(torch.tensor(original_audio).to(device), x_hat_tensor)
#     MSE_loss.append(loss_MSE.numpy())
#     loss_Mel = loss_mel(original_audio, x_hat_tensor.numpy())
#     Mel_loss.append(loss_Mel)
#
# np.savetxt('ast_MSEloss_4.txt', MSE_loss)
# np.savetxt("ast_MELloss_4.txt", Mel_loss)
