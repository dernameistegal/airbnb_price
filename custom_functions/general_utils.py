import torch
from sklearn.model_selection import train_test_split


def get_device(cuda_preference=True):
    print('cuda available:', torch.cuda.is_available(),
          '; cudnn available:', torch.backends.cudnn.is_available(),
          '; num devices:', torch.cuda.device_count())

    use_cuda = False if not cuda_preference else torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    device_name = torch.cuda.get_device_name(device) if use_cuda else 'cpu'
    print('Using device', device_name)
    return device


def train_val_test_split(listing_ids, train_size=0.7, random_state=42):
    ids_train, ids_val = train_test_split(listing_ids, train_size=train_size, random_state=random_state)
    ids_val, ids_test = train_test_split(ids_val, train_size=0.5, random_state=random_state)
    return ids_train, ids_val, ids_test
