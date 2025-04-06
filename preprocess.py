import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import os
import scipy
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader as DL
from torch.utils.data import TensorDataset as TData
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split as tts
import pickle

import zipfile
import pandas as pd
from sklearn.preprocessing import LabelEncoder

!unzip "LHNT_EEG.zip"

# applying a bandpass filter
def bandpass_filter(signal, crit_freq = [1, 40], sampling_freq = 125, plot = False, channel = 0):
  order = 4

  b, a = scipy.signal.butter(order, crit_freq, btype = 'bandpass', fs = sampling_freq)
  processed_signal = scipy.signal.filtfilt(b, a, signal, 1)

  if plot == True:
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel(f'Normalized amplitude of channel {channel}')
    plt.title(f'{crit_freq[0]}-{crit_freq[1]}Hz bandpass filter')
    signal_min = np.full((signal.shape[1], signal.shape[0]), np.min(signal, 1)).transpose()
    signal_max = np.full((signal.shape[1], signal.shape[0]), np.max(signal, 1)).transpose()
    normed_signal = (signal - signal_min) / (signal_max - signal_min)
    filtered_min = np.full((processed_signal.shape[1], processed_signal.shape[0]), np.min(processed_signal, 1)).transpose()
    filtered_max = np.full((processed_signal.shape[1], processed_signal.shape[0]), np.max(processed_signal, 1)).transpose()
    normed_filt = (processed_signal - filtered_min) / (filtered_max - filtered_min)
    plt.plot(np.arange(normed_signal[channel].size), normed_signal[channel], label = 'Input')
    plt.plot(np.arange(normed_filt[channel].size), normed_filt[channel], label = 'Transformed')
    plt.legend()

  return processed_signal


# function to segment eeg data based on sampling freq(Hz), window_size(s), and window_shift(s)
def segmentation(signal, sampling_freq=125, window_size=1, window_shift=0.016):
  w_size = int(sampling_freq * window_size)
  w_shift = int(sampling_freq * window_shift)
  segments = []
  i = 0
  while i + w_size <= signal.shape[1]:
    segments.append(signal[:, i: i + w_size])
    i += w_shift
  return segments

def channel_rearrangment(sig, channel_order):
    channel_order = [channel - 1 for channel in channel_order]
    reindexed = np.zeros_like(sig)
    for i, ind in enumerate(channel_order):
        reindexed[i] = sig[ind]
    return reindexed

ordered_channels = [1, 9, 11, 3, 2, 12, 10, 4, 13, 5, 15, 7, 14, 16, 6, 8]

train_x, test_x, train_y, test_y = tts(np_data, labels, test_size = 0.25)
val_x, test_x = test_x[:len(test_x)//2], test_x[len(test_x)//2:]
val_y, test_y = test_y[:len(test_y)//2], test_y[len(test_y)//2:]

# applying all preprocessing steps to create train and test data
train_eeg = []
train_labels = []
valid_eeg = []
valid_labels = []
test_eeg = []
test_labels = []
for sig, label in zip(train_x, train_y):
  if sig.shape[1] == 0: # excluding empty sample elements
    #print(name)
    continue
  reindexed_signal = channel_rearrangment(sig, ordered_channels)
  filtered_sig = bandpass_filter(reindexed_signal, [5, 40], 125) # bandpass filter
  normed_sig = (filtered_sig - np.mean(filtered_sig, 1, keepdims=True)) / np.std(filtered_sig, 1, keepdims=True) # standard scaling
  if np.isnan(normed_sig).any(): # excluding sample elements with nans
    print("nan")
    continue
  signals = segmentation(normed_sig, 125, window_size = 1.5, window_shift = 0.0175) # segmentation
  labels = [label] * len(signals)
  train_eeg.extend(signals)
  train_labels.extend(labels)

for sig, label in zip(val_x, val_y):
  if sig.shape[1] == 0: # excluding empty sample elements
    #print(name)
    continue
  reindexed_signal = channel_rearrangment(sig, ordered_channels)
  filtered_sig = bandpass_filter(reindexed_signal, [5, 40], 125) # bandpass filter
  normed_sig = (filtered_sig - np.mean(filtered_sig, 1, keepdims=True)) / np.std(filtered_sig, 1, keepdims=True) # standard scaling
  if np.isnan(normed_sig).any(): # excluding sample elements with nans
    print("nan")
    continue
  signals = segmentation(normed_sig, 125, window_size = 1.5, window_shift = 0.0175) # segmentation
  labels = [label] * len(signals)
  valid_eeg.extend(signals)
  valid_labels.extend(labels)

for sig, label in zip(test_x, test_y):
  if sig.shape[1] == 0: # excluding empty sample elements
    #print(name)
    continue
  reindexed_signal = channel_rearrangment(sig, ordered_channels)
  filtered_sig = bandpass_filter(reindexed_signal, [5, 40], 125) # bandpass filter
  normed_sig = (filtered_sig - np.mean(filtered_sig, 1, keepdims=True)) / np.std(filtered_sig, 1, keepdims=True) # standard scaling
  if np.isnan(normed_sig).any(): # excluding sample elements with nans
    print("nan")
    continue
  signals = segmentation(normed_sig, 125, window_size = 1.5, window_shift = 0.0175) # segmentation, changed to 125
  labels = [label] * len(signals)
  test_eeg.extend(signals)
  test_labels.extend(labels)

  columns_to_remove = [1, 2, 7, 8]
modified_train_eeg = []
modified_valid_eeg = []
modified_test_eeg = []
for array in train_eeg:
  modified_array = np.delete(array, columns_to_remove, axis=1)
  modified_train_eeg.append(modified_array)
for array in valid_eeg:
  modified_array = np.delete(array, columns_to_remove, axis=1)
  modified_valid_eeg.append(modified_array)
for array in test_eeg:
  modified_array = np.delete(array, columns_to_remove, axis=1)
  modified_test_eeg.append(modified_array)
train_eeg = modified_train_eeg
valid_eeg = modified_valid_eeg
test_eeg = modified_test_eeg

train_eeg_tensor = torch.zeros((len(train_eeg), train_eeg[0].shape[0], train_eeg[0].shape[1])) # untransposed dimensions 1 and 2
valid_eeg_tensor = torch.zeros((len(valid_eeg), valid_eeg[0].shape[0], valid_eeg[0].shape[1]))
test_eeg_tensor = torch.zeros((len(test_eeg), test_eeg[0].shape[0], test_eeg[0].shape[1]))
for i in range(len(train_eeg)):
  tens = torch.from_numpy(train_eeg[i].copy()) # no longer transposing before conversion to tensor
  train_eeg_tensor[i] = tens
for i in range(len(valid_eeg)):
  tens = torch.from_numpy(valid_eeg[i].copy())
  valid_eeg_tensor[i] = tens
for i in range(len(test_eeg)):
  tens = torch.from_numpy(test_eeg[i].copy())
  test_eeg_tensor[i] = tens
train_label_tensor = torch.zeros(len(train_labels), 2)
valid_label_tensor = torch.zeros(len(valid_labels), 2)
test_label_tensor = torch.zeros(len(test_labels), 2)
for i, val in enumerate(train_labels):
  train_label_tensor[i][val] = 1
for i, val in enumerate(valid_labels):
  valid_label_tensor[i][val] = 1
for i, val in enumerate(test_labels):
  test_label_tensor[i][val] = 1

train_ds = TData(train_eeg_tensor, train_label_tensor)
valid_ds = TData(valid_eeg_tensor, valid_label_tensor)
test_ds = TData(test_eeg_tensor, test_label_tensor)
train_dl = DL(train_ds, batch_size=64, shuffle= True, drop_last = True)
valid_dl = DL(valid_ds, batch_size=64, shuffle= True, drop_last = True)
test_dl = DL(test_ds, batch_size=64, shuffle = True, drop_last = True)

print(len(train_dl), len(valid_dl), len(test_dl))