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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

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
        
        mel = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)  # (64, 431) 44100
        mel = librosa.amplitude_to_db(np.abs(mel))

        mel = mel.T

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

    if val_loss <= min_loss_val:
        min_loss_val = val_loss
        best_model = model.state_dict()
        torch.save(best_model, 'pretrained_ast_LibriSpeech_stft_431_128.pt')

    ## Early stopping check
    if early_stop:
        if epoch < 30:
            continue
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
    
        if patience_counter >= patience:
            print(f'Early stopping after {epoch + 1} epochs without improvement.')
            break

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
