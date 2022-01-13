import torch
import torchvision
import os
import numpy as np
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


# dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath, ndata):
        self.filepath = filepath
        self.ndata = ndata
        self.filenames = os.listdir(self.filepath)[0:ndata]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, key):
        x = np.load(self.filepath + "/" + self.filenames[key])
        x = np.transpose(x, axes=[2, 0, 1])
        x = torch.from_numpy(x)
        return x


dataset = Dataset(hostpics_path, 10)
dataloader = torch.utils.data.DataLoader(dataset, bytch_size=1, shuffle=False)

vgg = torchvision.models.vgg19(pretrained=True)
eature_extractor = vgg.features[0:31]




# def compute_train_features(dataloader):
#     feature_extractor.to(device)
#     feature_extractor.eval()
#     out = torch.empty(501, 512, 16, 16)
#     start, stop = 0, 0
#
#     for batch in dataloader:
#         stop += len(batch["image"])
#         out[start:stop, ...] = feature_extractor(batch["image"].to(device)).detach()
#         start += len(batch["image"])
#
#     return out