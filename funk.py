from tools import *
from Dataset import BrainMRIDataset

def GetDataset(meta_data):

    train_meta_data = meta_data[:500]
    val_meta_data = meta_data[500:600].reset_index()

    transformation_train = T.Compose([T.RandomHorizontalFlip(),
                                      T.ToTensor()])

    transformation_val = T.Compose([T.ToTensor()])

    # PYTORCH TRANSFORMATION NOT SUPPORTED FOR 3D IMAGES

    train_dataset = BrainMRIDataset(train_meta_data)
    val_dataset = BrainMRIDataset(val_meta_data)

    loader_train = DataLoader(train_dataset,
                             batch_size=16,
                             shuffle=True)
    loader_val = DataLoader(val_dataset,
                            batch_size=16,
                            shuffle=True)

    return loader_train, loader_val

