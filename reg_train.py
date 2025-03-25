import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from mee_model import MEE
from tfe_model import TFE, TimeFrequencyCQT_Encoder
import time
import pickle
import yaml
from AudioDataset import norm_params, get_dataset, CurriculumDataset
import auraloss
from DECOMP import decompressor
import soundfile as sf
from sklearn.decomposition import PCA
import pandas as pd
from utiles import set_seed, SNRScheduler


def normSignal(x):
    x = x - torch.mean(x, dim=1, keepdim=True)
    x = x / torch.sqrt(torch.mean(torch.square(x), dim=1, keepdim=True))
    return x

def find_real_params(ranges, p):
    m = ranges[:, 0].reshape(1, -1)
    M = ranges[:, 1].reshape(1, -1)
    pp = (M - m) * p + m
    return pp

# Hyperparameters
seed = 1
set_seed(seed)
batch_size = 12
lr = 0.0001
epochs = 150
num_controls = 6
# control_ranges = np.array([[-60, 0],
#                            [0, 15],
#                            [0, 130],
#                            [0, 130],
#                            [0, 500],
#                            [0, 2000]])
control_ranges = np.array([[-38, 0],
                           [0, 7.3],
                           [0, 5],
                           [0, 5],
                           [0, 13],
                           [0, 705]])

loss_type = "params"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Data loading and preprocessing
audio_folder_path = "/data/hsun/Datasets/LibriSpeech/5profiles_large"
train_folder = "/data/hsun/Datasets/LibriSpeech/5profiles_large_train"
test_folder = "/data/hsun/Datasets/LibriSpeech/5profiles_large_test"
eval_folder = test_folder
output_folder = "/data/hsun/Datasets/LibriSpeech/5profiles_large_test_reg"
# audio_folder_path = "/data/hsun/Datasets/LibriSpeech/30profiles_large"
# train_folder = "/data/hsun/Datasets/LibriSpeech/30profiles_large_train"
# test_folder = "/data/hsun/Datasets/LibriSpeech/30profiles_large_test"
# output_folder = "/data/hsun/Datasets/LibriSpeech/30profiles_large_test_reg"

# compressors = pickle.load(open('30profiles_RMS.pkl', 'rb'))
compressors = {
    'A': {'param1': -32, 'param2': 3, 'param3': 5, 'param4': 5, 'param5': 13, 'param6': 435, 'param7': 2},
    'B': {'param1': -19.9, 'param2': 1.8, 'param3': 5, 'param4': 5, 'param5': 11, 'param6': 49, 'param7': 2},
    'C': {'param1': -24.4, 'param2': 3.2, 'param3': 5, 'param4': 5, 'param5': 5.8, 'param6': 112, 'param7': 2},
    'D': {'param1': -28.3, 'param2': 7.3, 'param3': 5, 'param4': 5, 'param5': 9, 'param6': 705, 'param7': 2},
    'E': {'param1': -38, 'param2': 4.9, 'param3': 5, 'param4': 5, 'param5': 3.1, 'param6': 257, 'param7': 2},
}

snr_scheduler = SNRScheduler(snr_start=40,
                             snr_step=5,
                             step_interval=20,
                             total_epochs=epochs,
                             min_snr=5,
                             schedule_type="linear")

train_dataset = get_dataset(train_folder, compressors)
train_dataset = CurriculumDataset(train_dataset, snr_scheduler)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          )

test_dataset = get_dataset(test_folder, compressors)
test_dataset = CurriculumDataset(test_dataset, snr_scheduler)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         )

Encoder = 'MEE'
if Encoder == 'MEE':
    config = yaml.full_load(open('MEE_configs.yaml', 'r'))
    model = MEE(config, num_controls)
elif Encoder == 'TFE':
    model = TFE(samplerate=16000, f_dim=64, t_dim=431, label_dim=num_controls)
    # model = TimeFrequencyCQT_Encoder()
else:
    raise ValueError("Not a correct encoder name !")

# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Number of trainable parameters : {pytorch_total_params}")
model = nn.DataParallel(model)
model = model.to(device)

fft_sizes = [256, 1024, 4096]
hop_sizes = [64, 256, 1024]
win_lengths = [256, 1024, 4096]

MRSTFT = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=fft_sizes,
    hop_sizes=hop_sizes,
    win_lengths=win_lengths,
    w_sc=1,
    device=device
).to(device)
MSE = torch.nn.MSELoss()
Mel = auraloss.freq.MelSTFTLoss(
    sample_rate=22050,
    fft_size=2048,
    hop_size=512,
    win_length=2048,
    n_mels=128,
    device=device,
    w_sc=0
).to(device)

def normSignal(x):
    x = x - torch.mean(x, dim=2, keepdim=True)
    x = x / torch.sqrt(torch.mean(torch.square(x), dim=2, keepdim=True))
    return x

def loss_params(q, q_hat, x, y, AFX):
    x_hat = AFX(y, q_hat)
    return 10 * MSE(q_hat, q), x_hat

def loss_mel(q, q_hat, x, y, AFX):
    x_hat = AFX(y, q_hat)
    # x = normSignal(x)cf
    # x_hat = normSignal(x_hat)
    return Mel(x_hat, x), x_hat


def get_lossFn(loss_type):
    if loss_type.lower() == 'params':
        return loss_params
    elif loss_type == 'mel':
        return loss_mel
    else:
        raise ValueError("Wrong loss type.")

### Training loop with early stopping
train_losses = []
val_losses = []

### Early stopping parameters
min_loss_val = 10
patience = 15  # Number of epochs with no improvement after which training will be stopped
best_val_loss = float('inf')
patience_counter = 0

# criterion = get_lossFn(loss_type)
criterion = MSE
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

torch.cuda.synchronize()
start = time.time()

for epoch in range(epochs):
    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    model.train()

    running_loss = 0.0
    correct_predictions_train = 0
    total_samples_train = 0
    all_labels_train = []
    all_predictions_train = []

    current_snr = snr_scheduler.get_current_snr()

    for batch_idx, (inputs, target, labels, real_q, norm_q, names) in enumerate(iter(train_loader)):
        inputs, norm_q = inputs.to(torch.float32), norm_q.to(torch.float32)
        inputs, norm_q = inputs.unsqueeze(1).to(device), norm_q.to(device)

        q_hat = model(inputs)
        # loss, estimate = criterion(
        #     q=q,
        #     q_hat=q_hat,
        #     x=inputs,
        #     y=target,
        #     AFX=AFX_Chain
        # )

        optimizer.zero_grad()
        loss = criterion(norm_q, q_hat)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    snr_scheduler.step()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, target, labels, real_q, norm_q, names) in enumerate(iter(test_loader)):
            inputs, norm_q = inputs.to(torch.float32), norm_q.to(torch.float32)
            inputs, norm_q = inputs.unsqueeze(1).to(device), norm_q.to(device)

            q_hat = model(inputs)
            # loss, estimate = criterion(
            #     q=q,
            #     q_hat=q_hat,
            #     x=estimate,
            #     y=target,
            #     AFX=AFX_Chain
            # )
            loss = criterion(norm_q, q_hat)
            val_loss += loss.item()
    val_loss /= len(test_loader)
    val_losses.append(val_loss)

    print(f'Epoch {epoch + 1:03d} | Current lr: {learning_rate:.7f} | Current SNR: {current_snr:.1f} dB \n Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    if val_loss <= min_loss_val:
        min_loss_val = val_loss
        best_model = model.state_dict()
        torch.save(best_model, 'pretrained_mee_LibriSpeech_5large_aug.pt')

    ### Early stopping check
    if epoch < patience:
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

print(f'Early stopping after {epoch + 1} epochs without improvement.')

best_val_loss = min(val_losses)
best_epoch = val_losses.index(best_val_loss) + 1
print(f'Best validation accuracy: {best_val_loss:.4f} achieved at epoch {best_epoch}')
print(f'Average training time: {time_per_epoch:.4f} h')


########
# eval #
########
# eval_losses_MSE = []
# eval_losses_Mel = []
#
# model.load_state_dict(torch.load('../pretrained models/pretrained_mee_LibriSpeech.pt', weights_only=True))
#
# torch.cuda.synchronize()
# tic = time.time()
#
# eval_loss_MSE = 0.0
# eval_loss_Mel = 0.0
# Estimated_Signals = []
# loss_MSE_all = []
# loss_Mel_all = []
# count = 0
#
# estimated_p = []
# real_labels = []
#
# q_tab = {
#     'name': [],
#     'label': [],
#     'real': [],
#     'estimated': [],
#     'q': []
# }
#
# eval_dataset = get_dataset(eval_folder, compressors)
# eval_loader = DataLoader(eval_dataset,
#                          batch_size=batch_size,
#                          shuffle=False,
#                          )
#
# model.eval()
# with torch.no_grad():
#     for batch_idx, (inputs, targets, labels, real_q, norm_q, names) in enumerate(iter(eval_loader)):
#         inputs, norm_q = inputs.to(torch.float32), norm_q.to(torch.float32)
#         inputs, norm_q = inputs.unsqueeze(1).to(device), norm_q.to(device)
#         q_hat = model(inputs)
#         # q_hat = find_real_params(control_ranges, q_hat.cpu().numpy())
#         estimated_p.append(q_hat)
#         real_labels.append(labels)
#
#         estimated_signals = []
#         for i in range(inputs.size(0)):
#             y = inputs[i]
#             x_real = targets[i]
#             real_label = labels[i]
#             real_parameter = real_q[i]
#             theta = q_hat[i]
#             audio_name = names[i]
#
#             q_tab['name'].append(audio_name)
#             q_tab['label'].append(real_label)
#             q_tab['real'].append(real_parameter.numpy())
#             q_tab['estimated'].append(find_real_params(control_ranges, theta.cpu().numpy()))
#             q_tab['q'].append(theta.cpu().numpy())
#
#             if real_label == 'O':
#                 estimated_signal = x_real
#             else:
#                 y = y.squeeze().cpu().numpy()
#                 parameters = find_real_params(control_ranges, theta.cpu().numpy())
#                 parameters = parameters.flatten()
#                 estimated_signal = decompressor(y, 16000, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], 2)
#
#             sf.write(output_folder + "/" + audio_name, estimated_signal, 16000)
#             # estimated_signals.append(estimated_signal)
#             # Estimated_Signals.append(np.array(estimated_signal))
#
#             count += 1
#             print(count)
#
#         # estimated_signals = torch.tensor(np.array(estimated_signals))
#         #
#         # loss_MSE = MSE(normSignal(estimated_signals), normSignal(targets))
#         # eval_loss_MSE += loss_MSE.item()
#         # loss_MSE_all.append(loss_MSE)
#         # loss_Mel = Mel(normSignal(estimated_signals.to(torch.float32)).unsqueeze(1),
#         #                normSignal(targets.to(torch.float32)).unsqueeze(1))
#         # eval_loss_Mel += loss_Mel.item()
#         # loss_Mel_all.append(loss_Mel)
#
#         torch.cuda.empty_cache()
#
#         if count > 500:
#             break
#
# df = pd.DataFrame(q_tab)
# df.to_excel("../results/q_LibriSpeech_5_noised_small.xlsx", index=False)


#######################
# Dimension reduction #
#######################

# p_hat = np.concatenate(estimated_p, axis=0)
# real_labels = np.concatenate(real_labels, axis=0)
#
# p_hat = np.vstack([p_hat, np.array([0,0,0,0,0,0])])
# real_labels = np.append(real_labels, '0')
#
# for i in range(30):
#     type_i = str(i+1)
#     real_labels = np.append(real_labels, type_i)
#     pp = list(compressors[type_i].values())
#     pp.pop()
#     pp = np.array(pp)
#     # pp = norm_params(control_ranges, pp)
#     p_hat = np.vstack([p_hat, pp])

#######
# PCA #
#######
import colorsys
from sklearn.decomposition import PCA

# def pca(X, k):  # k is the components you want
#     n_samples, n_features = X.shape
#     X = X - np.mean(X, axis=0)
#     # scatter matrix
#     scatter_matrix = np.dot(np.transpose(X), X)
#     # Calculate the eigenvectors and eigenvalues
#     eig_val, eig_vec = np.linalg.eig(scatter_matrix)
#     eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
#     # sort eig_vec based on eig_val from highest to lowest
#     eig_pairs.sort(reverse=True)
#     # select the top k eig_vec
#     feature = np.array([ele[1] for ele in eig_pairs[:k]])
#     # get new data
#     data = np.dot(X, np.transpose(feature))
#     # Total inertia is the sum of eigenvalues
#     total_inertia = np.sum(eig_val)
#     # Sort eigenvalues in descending order
#     sorted_eigenvalues = np.sort(eig_val)[::-1]
#     # Calculate explained inertia percentage
#     explained_inertia_percentage = sorted_eigenvalues / total_inertia * 100
#
#     return eig_vec, data, explained_inertia_percentage

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        # Use golden ratio to spread hues evenly
        hue = i * 0.618033988749895 % 1
        # Alternate saturation and value for better distinction
        saturation = 0.8 + (i % 2) * 0.2
        value = 0.9 if i % 3 == 0 else (0.7 if i % 3 == 1 else 0.5)
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return colors

# pca = PCA(n_components=3)
# p_reduced = pca.fit_transform(p_hat)
# inertia = pca.explained_variance_ratio_  # array([0.95199446, 0.04185209])
# eigen_vec, p_reduced, inertia = pca(p_hat, 3)  # array([9.51994462e+01, 4.18520937e+00, 3.85317607e-01, 1.75760684e-01, 4.96441419e-02, 4.62202846e-03])


########### 2D plot
# colors = generate_distinct_colors(31)
# unique_labels = sorted(list(set(real_labels)))
# color_dict = dict(zip(unique_labels, colors))
#
# plt.figure(figsize=(12, 8))
# for label in unique_labels:
#     mask = real_labels == label
#     plt.scatter(p_reduced[mask, 0], p_reduced[mask, 1], c=[color_dict[label]])
# last_31_indices = range(len(p_reduced)-31, len(p_reduced))
# for idx in last_31_indices:
#     plt.annotate(real_labels[idx],
#                 (p_reduced[idx, 0], p_reduced[idx, 1]),
#                 xytext=(5, 5),
#                 textcoords='offset points',
#                 fontsize=8,
#                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
# plt.title('PCA of estimated DRC parameters')
# plt.xlabel('P1')
# plt.ylabel('P2')
# plt.show(block=True)

############# 3D plot
# fig = plt.figure(figsize=(16, 10))
# ax = fig.add_subplot(111, projection='3d')
#
# # Create a mapping of labels to colors
# unique_labels = sorted(list(set(real_labels)))
# color_dict = dict(zip(unique_labels, colors))
#
# # Plot points for each unique label
# for label in unique_labels:
#     mask = real_labels == label
#     ax.scatter(p_reduced[mask, 0], p_reduced[mask, 1], p_reduced[mask, 2],
#                c=[color_dict[label]],
#                label=label,
#                alpha=0.7,
#                s=50)
#
# last_31_indices = range(len(p_reduced)-31, len(p_reduced))
# for idx in last_31_indices:
#     ax.text(p_reduced[idx, 0], p_reduced[idx, 1], p_reduced[idx, 2],
#             real_labels[idx],
#             color='black',
#             fontsize=8,
#             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
# ax.set_title('PCA of estimated DRC parameters')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.tight_layout()
# plt.show(block=True)


#######
# MDS #
#######
# from sklearn.manifold import MDS
# from audio_loss import loss_mse, MelSTFTLoss
#
# def mse_metrix(X):
#     n_samples = X.shape[0]
#     distance_matrix = np.zeros((n_samples, n_samples))
#
#     for i in range(n_samples):
#         for j in range(n_samples):
#             distance_matrix[i, j] = loss_mse(X[i], X[j])
#
#     return distance_matrix
#
# def mel_metrix(X):
#     n_samples = X.shape[0]
#     distance_matrix = np.zeros((n_samples, n_samples))
#
#     Mel = MelSTFTLoss()
#     for i in range(n_samples):
#         for j in range(n_samples):
#             distance_matrix[i, j] = Mel.compute_loss(X[i], X[j])
#
#     return distance_matrix
#
# class MyMDS:
#     def __init__(self, n_components):
#         self.n_components = n_components
#
#     def fit(self, data):
#         m, n = data.shape
#         dist = np.zeros((m, m))
#         disti = np.zeros(m)
#         distj = np.zeros(m)
#         B = np.zeros((m, m))
#         for i in range(m):
#             dist[i] = np.sum(np.square(data[i] - data), axis=1).reshape(1, m)
#         for i in range(m):
#             disti[i] = np.mean(dist[i, :])
#             distj[i] = np.mean(dist[:, i])
#         distij = np.mean(dist)
#         for i in range(m):
#             for j in range(m):
#                 B[i, j] = -0.5 * (dist[i, j] - disti[i] - distj[j] + distij)
#         lamda, V = np.linalg.eigh(B)
#         index = np.argsort(-lamda)[:self.n_components]
#         diag_lamda = np.sqrt(np.diag(-np.sort(-lamda)[:self.n_components]))
#         V_selected = V[:, index]
#         Z = V_selected.dot(diag_lamda)
#
#
# signal_all = np.loadtxt("modified_signal.txt")
# # signal_all = signal_all[1:61]
# audio_labels = []
#
# aaa = np.ones(1830, dtype=bool)
# aaa[::61] = False
# signal_all = signal_all[aaa]
#
# mds = MDS(n_components=2, dissimilarity='precomputed')
# mel_distance_matrix = mel_metrix(signal_all)
# audio_mds = mds.fit_transform(mel_distance_matrix)
#
# parameter_names = ['L', 'R', 'v_att', 'v_rel', 'g_att', 'g_rel']
# ref_labels = [element for element in parameter_names for _ in range(10)]
# ref_labels = np.array(ref_labels)
# ref_labels = np.tile(ref_labels, 30)
#
# colors = generate_distinct_colors(6)
# unique_labels = sorted(list(set(ref_labels)))
# color_dict = dict(zip(unique_labels, colors))
#
# ############# 2D plot
# plt.figure(figsize=(12, 8))
# for label in unique_labels:
#     mask = ref_labels == label
#     plt.scatter(audio_mds[mask, 0], audio_mds[mask, 1], c=[color_dict[label]], label=label)
# plt.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('MDS with L_Mel')
# plt.xlabel('P1')
# plt.ylabel('P2')
# plt.tight_layout()
# plt.show(block=True)

############### 3D plot
# fig = plt.figure(figsize=(16, 10))
# ax = fig.add_subplot(111, projection='3d')
# for label in unique_labels:
#     mask = ref_labels == label
#     ax.scatter(audio_mds[mask, 0], audio_mds[mask, 1], audio_mds[mask, 2],
#                c=[color_dict[label]],
#                label=label,
#                alpha=0.7,
#                s=50)
# plt.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')
# ax.set_title('MDS (3D) with L_MSE')
# ax.set_xlabel('P1')
# ax.set_ylabel('P2')
# ax.set_zlabel('P3')
# plt.tight_layout()
# plt.show(block=True)


########
# TSNE #
########
# from sklearn.manifold import TSNE
#
# tsne = TSNE(n_components=3, perplexity=30.0, init='pca')
# p_reduced = tsne.fit_transform(p_hat)
#
# colors = generate_distinct_colors(31)
# unique_labels = sorted(list(set(real_labels)))
# color_dict = dict(zip(unique_labels, colors))
#
# ########## 2D plot
#
# plt.figure(figsize=(12, 8))
# for label in unique_labels:
#     mask = real_labels == label
#     plt.scatter(p_reduced[mask, 0], p_reduced[mask, 1], c=[color_dict[label]])
# last_31_indices = range(len(p_reduced)-31, len(p_reduced))
# for idx in last_31_indices:
#     plt.annotate(real_labels[idx],
#                 (p_reduced[idx, 0], p_reduced[idx, 1]),
#                 xytext=(5, 5),
#                 textcoords='offset points',
#                 fontsize=8,
#                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
# plt.title('t-SNE of estimated DRC parameters')
# plt.xlabel('P1')
# plt.ylabel('P2')
# plt.show(block=True)

############ 3D plot
# fig = plt.figure(figsize=(16, 10))
# ax = fig.add_subplot(111, projection='3d')
#
# # Create a mapping of labels to colors
# unique_labels = sorted(list(set(real_labels)))
# color_dict = dict(zip(unique_labels, colors))
#
# # Plot points for each unique label
# for label in unique_labels:
#     mask = real_labels == label
#     ax.scatter(p_reduced[mask, 0], p_reduced[mask, 1], p_reduced[mask, 2],
#                c=[color_dict[label]],
#                label=label,
#                alpha=0.7,
#                s=50)
#
# last_31_indices = range(len(p_reduced)-31, len(p_reduced))
# for idx in last_31_indices:
#     ax.text(p_reduced[idx, 0], p_reduced[idx, 1], p_reduced[idx, 2],
#             real_labels[idx],
#             color='black',
#             fontsize=8,
#             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
# ax.set_title('t-SNE of estimated DRC parameters')
# ax.set_xlabel('P1')
# ax.set_ylabel('P2')
# ax.set_zlabel('P3')
# plt.tight_layout()
# plt.show(block=True)