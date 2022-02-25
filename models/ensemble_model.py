import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class EnsembleModel2(nn.Module):

    def __init__(self, no_of_thumb, no_of_desc, no_of_rev, no_of_cont, cat_emb_dims, lin_layer_sizes, thumb_dropout,
                 desc_dropout, rev_dropout, cont_dropout, cat_dropout, linear_layer_dropout, bn_layers=False, only_unstructured=False):
        super().__init__()

        # number of features for different feature types
        self.no_of_thumb = no_of_thumb
        self.no_of_desc = no_of_desc
        self.no_of_rev = no_of_rev
        self.no_of_cont = no_of_cont
        self.use_bn_layers = bn_layers

        if len(cat_emb_dims) == 0:
            self.no_of_cat = 0
        else:
            self.no_of_cat = np.sum([y for x, y in cat_emb_dims])

        # Embedding layers
        if not len(cat_emb_dims) == 0:
            self.cat_emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in cat_emb_dims])
        else:
            self.cat_emb_layers = []

        # Linear Layers
        first_lin_layer = nn.Linear(
            self.no_of_thumb + self.no_of_desc + self.no_of_rev + self.no_of_cont + self.no_of_cat,
            lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] +
                                        [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                                         for i in range(len(lin_layer_sizes) - 1)])
        if only_unstructured:
            for i, module in enumerate(self.lin_layers):
                if i == 0:
                    module.bias.data.fill_(3)
                    module.weight.data = torch.eye(422)
                else:
                    module.bias.data.fill_(0)
                    module.weight.data = torch.eye(422)

        self.last_lin_layer = nn.Linear(lin_layer_sizes[-1], 1)

        # for lin_layer in self.lin_layers:
        #     nn.init.kaiming_normal_(lin_layer.weight.data)
        # nn.init.kaiming_normal_(self.last_lin_layer.weight.data)

        # Batch Norm Layers
        self.bn_cont_features = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size)
                                        for size in lin_layer_sizes])

        # Dropout Layers (can be modified by using dropout rate as argument)
        self.thumb_dropout = nn.Dropout(thumb_dropout)
        self.desc_dropout = nn.Dropout(desc_dropout)
        self.rev_dropout = nn.Dropout(rev_dropout)
        self.cont_dropout = nn.Dropout(cont_dropout)
        self.cat_dropout = nn.Dropout(cat_dropout)
        self.linear_droput_layers = nn.ModuleList([nn.Dropout(dropout_rate) for dropout_rate in linear_layer_dropout])

    def forward(self, thumb_data, desc_data, rev_data, cont_data, cat_data):

        # generate embeddings and apply dropout
        if not len(self.cat_emb_layers) == 0:
            cat_data = [self.cat_dropout(cat_emb_layer(cat_data[:, i]))
                        for i, cat_emb_layer in enumerate(self.cat_emb_layers)]

            cat_data = torch.cat(cat_data, 1)

        # dropout on precomputed embeddings and batchnorm on continuous features which were not normalized yet
        thumb_data = self.thumb_dropout(thumb_data)
        desc_data = self.desc_dropout(desc_data)
        rev_data = self.rev_dropout(rev_data)
        cont_data = self.cont_dropout(cont_data)
        cont_data = torch.cat([thumb_data, desc_data, rev_data, cont_data], dim=1)

        if not len(self.cat_emb_layers) == 0:
            x = torch.cat([cont_data, cat_data], dim=1)
        else:
            x = cont_data

        if self.use_bn_layers:
            for lin_layer, dropout_layer, bn_layer in \
                    zip(self.lin_layers, self.linear_droput_layers, self.bn_layers):
                x = bn_layer(F.relu(lin_layer(x)))
                x = dropout_layer(x)

            x = self.last_lin_layer(x)

        else:
            for lin_layer, dropout_layer in \
                    zip(self.lin_layers, self.linear_droput_layers):
                x = F.relu(lin_layer(x))
                x = dropout_layer(x)

            x = self.last_lin_layer(x)

        return x

    def generate_cat_embeddings(self, variable_number, category_number):

        with torch.no_grad():
            category_number = torch.tensor(category_number)
            category_number = torch.unsqueeze(category_number, 0)
            category_number = torch.unsqueeze(category_number, 0)

            embedding = self.cat_emb_layers[variable_number](category_number)
            embedding = torch.squeeze(embedding)

        return embedding


class EnsembleDataset2(Dataset):
    def __init__(self, data, thumb_col, desc_col, rev_col, cat_cols, output_col):
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
        self.desc_X = torch.from_numpy(self.desc_X).float()

        self.rev_X = data[self.rev_col].values.reshape(-1, 1)
        self.rev_X = np.apply_along_axis(np.concatenate, 1, self.rev_X)
        self.rev_X = torch.from_numpy(self.rev_X).float()

        self.thumb_X = data[self.thumb_col].values.reshape(-1, 1)
        self.thumb_X = np.apply_along_axis(np.concatenate, 1, self.thumb_X)
        self.thumb_X = torch.from_numpy(self.thumb_X).float()

        self.cont_X = data[self.cont_cols].values
        self.cont_X = torch.from_numpy(self.cont_X.astype(np.float32)).float()

        if not len(self.cat_cols) == 0:
            self.cat_X = data[self.cat_cols].values
            self.cat_X = torch.from_numpy(self.cat_X.astype(np.float32)).float()
        else:
            self.cat_X = torch.clone(self.cont_X)
            self.cat_X[...] = 0

        self.output = data[output_col].values.reshape(-1, 1)
        self.output = torch.from_numpy(self.output).float()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.thumb_X[idx], self.desc_X[idx], self.rev_X[idx], self.cont_X[idx], self.cat_X[idx], self.output[idx]