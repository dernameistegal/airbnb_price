import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    def __init__(self, nfeatures):
        super(EnsembleModel, self).__init__()

        self.linear1 = nn.Linear(300 + nfeatures, 300)
        self.linear2 = nn.Linear(300, 200)
        self.linear3 = nn.Linear(200, 100)
        self.linear4 = nn.Linear(100, 1)

    def forward(self, pic_embedding, description_embedding, reviews_embedding, features):
        embeddings = torch.hstack((pic_embedding, description_embedding, reviews_embedding, features))

        x = self.linear1(embeddings)
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))
        x = self.linear4(F.relu(x))

        return x
