import torch
import os
import numpy as np


def calculate_channelwise_moments(data_dir):
    data_paths = os.listdir(data_dir)

    means = np.empty((len(data_paths), 3))
    stds = np.empty((len(data_paths), 3))

    for i in range(len(data_paths)):
        temp = np.load(data_dir + "/" + data_paths[i])
        temp /= 255

        means[i] = np.mean(temp, dim=(0,1))
        stds[i] = np.std(temp, axis=(0,1))

    means = np.mean(means, dim=0)
    std = np.mean(stds, dim=0)
    
    return means, std


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath, channelmeans, channelstds, ndata):
        self.filepath = filepath
        self.ndata = ndata
        self.filenames = os.listdir(self.filepath)[0:ndata]
        self.channelmeans = channelmeans
        self.channelstds = channelstds

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, key):
        x = np.load(self.filepath + "/" + self.filenames[key])
        x = np.transpose(x, axes=[2, 0, 1])
        x /= 255
        x -= self.channelmeans
        x /= self.channelstds
        x = torch.from_numpy(x)

        return x


def compute_train_features(device, dataloader, feature_extractor, output_dim=[512, 16, 16]):
    feature_extractor.to(device)
    feature_extractor.eval()

    ndata = len(dataloader.dataset)
    output = torch.empty(ndata, output_dim[0], output_dim[1], output_dim[2])
    start, stop = 0, 0

    with torch.no_grad:
        for batch in dataloader:
            stop += len(batch)
            output[start:stop, ...] = feature_extractor(batch.to(device))
            start += len(batch)

    return output


if __name__ == "main":
    import torchvision
    import json
    import pandas as pd

    # paths
    root_dir = "G:/.shortcut-targets-by-id/1j3VRcg8GosML98qgJcN3di-lgTFtcd0n/data/"
    hostpics_path = root_dir + "hostpics"

    # missing_data_path = root_dir + "missing_data.json"
    # listings_meta_path = root_dir + "data1/listings.csv.gz"
    # listings_path = root_dir + "/data1/listings.csv"

    # data
    # listings_meta = pd.read_csv(listings_meta_path)
    # listings = pd.read_csv(listings_path)
    #
    # with open(missing_data_path, "r") as temp:
    #     missing_data = json.load(temp)

    # initialize dataset and dataloader
    dataset = Dataset(hostpics_path, 10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # get pretrained model
    vgg = torchvision.models.vgg19(pretrained=True)
    feature_extractor = vgg.features[0:31]

    # compute features for later training
    device = "cpu"
    compute_train_features(device, dataloader, feature_extractor)
