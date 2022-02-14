import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_device(cuda_preference=True):
    print('cuda available:', torch.cuda.is_available(),
          '; cudnn available:', torch.backends.cudnn.is_available(),
          '; num devices:', torch.cuda.device_count())

    use_cuda = False if not cuda_preference else torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    device_name = torch.cuda.get_device_name(device) if use_cuda else 'cpu'
    print('Using device', device_name)
    return device


def train_val_test_split(listing_ids, train_size=0.7, random_state=42):
    ids_train, ids_val = train_test_split(listing_ids, train_size=train_size, random_state=random_state)
    ids_val, ids_test = train_test_split(ids_val, train_size=0.5, random_state=random_state)
    return ids_train, ids_val, ids_test


def plot(title, label, training, validation, yscale='linear', legend=["Training", "Validation"],
         thinning=1, save_path=None, size=(5,4)):

    training = training[::thinning]
    validation = validation[::thinning]
    epoch_array = epoch_array[::thinning]
    plt.figure(figsize=size)

    sns.set(style='ticks')
    plt.plot(epoch_array, training,linestyle='dashed', marker='o', zorder=-1)
    plt.plot(epoch_array, validation, linestyle='dashed', marker='o', zorder=-1)
    legend = [legend[0], legend[1]]
    plt.legend(legend)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.yscale(yscale)
    plt.title(title, fontsize=15)

    if save_path:
        plt.savefig(str(save_path), bbox_inches='tight', format="svg", transparent=True)
    plt.show()
