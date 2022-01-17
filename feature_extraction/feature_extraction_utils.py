import torch
import os
import numpy as np
from tqdm import tqdm


def calculate_channelwise_moments(data_dir):
    data_paths = os.listdir(data_dir)

    means = np.empty((len(data_paths), 3))
    stds = np.empty((len(data_paths), 3))

    for i in tqdm(range(len(data_paths))):
        temp = np.load(data_dir + "/" + data_paths[i])
        temp = temp.astype(float)
        temp /= 255

        means[i] = np.mean(temp, axis=(0,1))
        stds[i] = np.std(temp, axis=(0,1))

    means = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return means, std


class Dataset(torch.utils.data.Dataset):
    def __init__(self, picture_path, response_path, channel_moments, ndata):
        self.ndata = ndata

        self.picture_path = picture_path
        self.response_path = response_path
        self.picture_names = os.listdir(self.picture_path)[0:ndata]
        self.response_names = os.listdir(self.response_path)[0:ndata]

        self.channelmeans = channel_moments[:, 0]
        self.channelstds = channel_moments[:, 1]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, key):
        x = np.load(self.picture_path + "/" + self.picture_names[key])

        # transform (to tensor and normalize)
        x = torch.from_numpy(x).type(torch.FloatTensor)
        x /= 255
        x -= self.channelmeans
        x /= self.channelstds
        x = torch.permute(x, dims=[2, 0, 1])

        y = np.load(self.response_path + "/" + self.picture_names[key])
        y = torch.from_numpy(x).type(torch.FloatTensor)

        return x, y


def compute_train_features(device, dataloader, feature_extractor):
    feature_extractor.to(device)
    feature_extractor.eval()

    output = []
    start, stop = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            output.append(feature_extractor(batch.to(device)))

    output = torch.cat(output, dim=0)

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
