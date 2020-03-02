from funk import *
from CNN import CNN3D
# from CNNUeda import CNN3Dueda

if __name__ == "__main__":

    meta_data_train = pd.read_csv(data_dir + 'meta/meta_data_reg_train.csv')
    meta_data_test = pd.read_csv(data_dir + 'meta/meta_data_reg_test.csv')
    loader_train, loader_val, loader_test = GetDataset(meta_data_train, meta_data_test, chunk=False)

    CNN = CNN3D(chunk=False)
    CNN.train(loader_train, loader_val, loader_test)

    # CNN = CNN3Dueda()
    # CNN.train(loader_train, loader_val, loader_test)