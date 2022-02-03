import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):
    def __init__(self, picture_dir, response_dir, channel_moments, ndata):
        self.ndata = ndata

        self.picture_dir = picture_dir
        self.response_dir = response_dir

        self.picture_names = os.listdir(self.picture_dir)
        self.picture_names = self.picture_names[0:ndata]

        self.channelmeans = channel_moments[:, 0]
        self.channelstds = channel_moments[:, 1]

    def __len__(self):
        return len(self.picture_names)

    def __getitem__(self, key):
        x = np.load(self.picture_dir + "/" + self.picture_names[key])

        # transform (to tensor and normalize)
        x = torch.from_numpy(x).type(torch.FloatTensor)
        x /= 255
        # x -= self.channelmeans
        x -= torch.tensor([0.485, 0.456, 0.406])
        # x /= self.channelstds
        x /= torch.tensor([0.229, 0.224, 0.225])
        x = torch.permute(x, dims=[2, 0, 1])

        y = np.load(self.response_dir + "/" + self.picture_names[key])
        y = torch.from_numpy(y).type(torch.FloatTensor)
        y = torch.log(y)

        return x, y
