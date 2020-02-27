from funk import *
from CNN import CNN3D

if __name__ == "__main__":

    meta_data_all = pd.read_csv(data_dir + 'meta/meta_data_all.csv')
    loader_train, loader_val = GetDataset(meta_data_all)
    CNN = CNN3D()
    CNN.train(loader_train, loader_val)