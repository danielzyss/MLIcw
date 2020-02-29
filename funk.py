from tools import *
from Dataset import BrainMRIDataset
from torch.utils.data import random_split

def GetDataset(meta_data_train, meta_data_test):

    transformation_train = T.Compose([T.RandomHorizontalFlip(),
                                      T.ToTensor()])

    transformation_val = T.Compose([T.ToTensor()])

    # PYTORCH TRANSFORMATION NOT SUPPORTED FOR 3D IMAGES

    train_dataset = BrainMRIDataset(meta_data_train)
    test_dataset = BrainMRIDataset(meta_data_train)

    train_dataset, val_dataset = random_split(train_dataset, [400, 100])

    loader_train = DataLoader(train_dataset,
                             batch_size=16,
                             shuffle=True)
    loader_val = DataLoader(val_dataset,
                            batch_size=16,
                            shuffle=True,)
    loader_test = DataLoader(test_dataset,
                            batch_size=16,
                            shuffle=True,)

    return loader_train, loader_val, loader_test

