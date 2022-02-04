import torch
import os
import numpy as np


class ThumbnailsDataset(torch.utils.data.Dataset):
    def __init__(self, picture_dir, response_dir, split):

        self.split = ["thumbnail" + index + ".npy" for index in split]
        self.picture_dir = picture_dir
        self.response_dir = response_dir

    def __len__(self):
        return len(self.split)

    def __getitem__(self, key):
        x = np.load(self.picture_dir + "/" + self.split[key])

        # transform (to FloatTensor and normalize)
        x = torch.from_numpy(x).type(torch.FloatTensor)
        x /= 255
        x -= torch.tensor([0.485, 0.456, 0.406])
        x /= torch.tensor([0.229, 0.224, 0.225])
        x = torch.permute(x, dims=[2, 0, 1])

        y = np.load(self.response_dir + "/" + self.split[key])
        y = torch.from_numpy(y).type(torch.FloatTensor)
        y = torch.log(y)

        return x, y
