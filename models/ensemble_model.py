import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class EnsembleModel(nn.Module):
    def __init__(self, nfeatures):
        super(EnsembleModel, self).__init__()

        self.linear1 = nn.Linear(300 + nfeatures, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 1)

    def forward(self, pic_embedding, description_embedding, reviews_embedding, features):
        embeddings = torch.hstack((pic_embedding, description_embedding, reviews_embedding, features))

        x = self.linear1(embeddings)
        x = self.linear2(F.relu(x))
        x = self.linear3(x)

        return x


# requires label encoding for categories
class EnsembleModel2(nn.Module):

    def __init__(self, no_of_thumb=100, no_of_desc=100, no_of_rev=100, no_of_cont, cat_emb_dims, lin_layer_sizes):
        super().__init__()

        # number of features for different feature types
        self.no_of_thumb = no_of_thumb
        self.no_of_desc = no_of_desc
        self.no_of_rev = no_of_rev
        self.no_of_cont = no_of_cont
        self.no_of_cat = np.sum([y for x, y in cat_emb_dims])

        # Embedding layers
        self.cat_emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                             for x, y in cat_emb_dims])
        # Linear Layers
        first_lin_layer = nn.Linear(
            self.no_of_thumb + self.no_of_desc + self.no_of_rev + self.no_of_cont + self.no_of_cat,
            lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] +
                                        [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                                         for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        self.last_lin_layer = nn.Linear(lin_layer_sizes[-1], 1)
        nn.init.kaiming_normal_(self.last_lin_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_thumb + self.no_of_desc + self.no_of_rev, self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size)
                                        for size in lin_layer_sizes])

        # Dropout Layers (can be modified by using dropout rate as argument)
        self.thumb_dropout = nn.Dropout(0.5)
        self.desc_dropout = nn.Dropout(0.5)
        self.rev_dropout = nn.Dropout(0.5)
        self.cat_dropout_layer = nn.Dropout(0.5)
        self.linear_droput_layers = nn.ModuleList([nn.Dropout(0.5)] * len(lin_layer_sizes))

    def forward(self, thumb_data, desc_data, rev_data, cont_data, cat_data):

        # generate embeddings and apply dropout
        x = [cat_emb_layer(cat_data[:, i])
             for i, cat_emb_layer in enumerate(self.cat_emb_layers)]

        x = torch.cat(x, 1)
        x = self.emb_dropout_layer(x)

        # normalize data of other features and apply dropout
        cont_data = torch.cat([thumb_data, desc_data, rev_data, cont_data], dim=1)
        normalized_cont_data = self.first_bn_layer(cont_data)

        x = torch.cat([x, normalized_cont_data], dim=1)

        for lin_layer, dropout_layer, bn_layer in \
                zip(self.lin_layers, self.droput_layers, self.bn_layers):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x


class EnsembleDataset2(Dataset):
    def __init__(self, data, desc_col, rev_col, thumb_col, cat_cols, output_col):
        self.length = len(data)

        # lists of columns that belong to predictor type (category, continuous, description, ...)
        self.desc_col = desc_col
        self.rev_col = rev_col
        self.thumb_col = thumb_col
        self.cat_cols = cat_cols
        self.output_col = output_col
        self.cont_cols = [col for col in data.columns
                          if col not in (self.cat_cols + self.desc_col + self.rev_col + self.thumb_col + self.output_col)]

        # actual data that belongs to predictors
        self.desc_X = data[self.desc_col].values.reshape(-1, 1)
        self.desc_X = np.apply_along_axis(np.concatenate, 1, self.desc_X)
        self.desc_X = torch.from_numpy(self.desc_X.astype(np.float64))

        self.rev_X = data[self.rev_col].values.reshape(-1, 1)
        self.rev_X = np.apply_along_axis(np.concatenate, 1, self.rev_X)
        self.rev_X = torch.from_numpy(self.rev_X.astype(np.float64))

        self.thumb_X = data[self.thumb_col].values.reshape(-1, 1)
        self.thumb_X = np.apply_along_axis(np.concatenate, 1, self.thumb_X)
        self.thumb_X = torch.from_numpy(self.thumb_X.astype(np.float64))

        self.cont_X = data[self.cont_cols].values
        self.cont_X = torch.from_numpy(self.cont_X.astype(np.float64))

        self.cat_X = data[self.cat_cols].values
        self.cat_X = torch.from_numpy(self.cat_X.astype(np.int64))

        self.output = data[output_col].values.reshape(-1, 1)
        self.output = torch.from_numpy(self.output.astype(np.float64))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.thumb_X[idx], self.desc_X[idx], self.rev_X[idx], self.cont_X[idx], self.cat_X[idx], self.output[idx]