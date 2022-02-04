import torch
import os
import numpy as np


class ThumbnailsDataset(torch.utils.data.Dataset):
    def __init__(self, thumbnail_dir, response, split):

        self.split = split
        self.thumbnail_dir = thumbnail_dir
        self.response = response

    def __len__(self):
        return len(self.split)

    def __getitem__(self, key):
        x = np.load(self.thumbnail_dir + "/thumbnail" + str(self.split[key]) + ".npy")

        # transform (to FloatTensor and normalize)
        x = torch.from_numpy(x).type(torch.FloatTensor)
        x /= 255
        x -= torch.tensor([0.485, 0.456, 0.406])
        x /= torch.tensor([0.229, 0.224, 0.225])
        x = torch.permute(x, dims=[2, 0, 1])

        y = self.response[self.split[key]]
        y = torch.from_numpy(y).type(torch.FloatTensor)
        y = torch.log(y)

        return x, y
