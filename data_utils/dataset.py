from torch.utils.data import Dataset
import pandas as pd


class dataset(Dataset):
    def __init__(self, datapath, columns):
        data = pd.read_pickle(datapath)
        super().__init__()
        self.data = data
        self.columns = columns

    def __getitem__(self, key):
        return self.data["log_price"].iloc[key], self.data[self.columns].iloc[key], self.data.index[key]

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    """
    custom collate fn
    :param batch: batch input from dataloader
    :return: price a tensor, columns a pandas dataframe, ids a tensor
    """
    price, columns, index = list(zip(*batch))
    price, index = torch.tensor(price), torch.tensor(index)
    columns = pd.concat(columns)
    return price, columns, ids