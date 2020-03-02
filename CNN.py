from parameters import *


class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()

        self.conv11 = Conv3d(in_channels, 120 , kernel_size=1, stride=1)
        self.batchnorm11 = nn.BatchNorm3d(120)
        self.conv12 = Conv3d(in_channels, 120, kernel_size=1, stride=1)
        self.batchnorm12 = nn.BatchNorm3d(120)
        self.conv13 = Conv3d(in_channels, 120, kernel_size=1, stride=1)
        self.batchnorm13 = nn.BatchNorm3d(120)
        self.maxpool14 = MaxPool3d(kernel_size=3, padding=1, stride=1)

        self.conv22 = Conv3d(120, 120, kernel_size=3, padding=1)
        self.batchnorm22 = nn.BatchNorm3d(120)
        self.conv23 = Conv3d(120, 120, kernel_size=3, padding=1)
        self.batchnorm23 = nn.BatchNorm3d(120)
        self.conv24 = Conv3d(in_channels, 120, kernel_size=1)
        self.batchnorm24 = nn.BatchNorm3d(120)

        self.relu = nn.ReLU()

    def forward(self,x):

        x11 = self.relu(self.batchnorm11(self.conv11(x)))
        x12 = self.relu(self.batchnorm12(self.conv12(x)))
        x13 = self.relu(self.batchnorm13(self.conv13(x)))
        x14 = self.maxpool14(x)


        x22 = self.relu(self.batchnorm22(self.conv22(x12)))
        x23 = self.relu(self.batchnorm23(self.conv23(x13)))
        x24 = self.relu(self.batchnorm24(self.conv24(x14)))

        output = torch.cat((x11, x22, x23, x24), 1)

        return output

class FireModule(nn.Module):
    def __init__(self, in_channels):
        super(FireModule, self).__init__()

        self.conv1 = Conv3d(in_channels, 128, kernel_size=1)
        self.batchnorm1 = nn.BatchNorm3d(128)

        self.conv21 = Conv3d(128, 256, kernel_size=3, stride=(2,2,2), padding=1)
        self.batchnorm21 = nn.BatchNorm3d(256)
        self.conv22 = Conv3d(128, 256, kernel_size=1, stride=(2,2,2))

        self.relu = nn.ReLU()

    def forward(self, x):

        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x21 = self.relu(self.batchnorm21(self.conv21(x1)))
        x22 = self.relu(self.conv22(x1))

        output = torch.cat((x21, x22), 1)

        return output

class CNNModel(nn.Module):

    def __init__(self, chunk=True):
        super(CNNModel, self).__init__()

        self.chunk = chunk


        if self.chunk:
            self.stem = nn.Sequential(Conv3d(1,48, kernel_size=3, stride=(2,2,1), bias=True, padding=2),
                                      nn.BatchNorm3d(48),
                                      nn.ReLU(),
                                      MaxPool3d(kernel_size=3, stride=(2,2,1)),
                                      Conv3d(48,96, kernel_size=3, bias=True, padding=2),
                                      nn.BatchNorm3d(96),
                                      nn.ReLU(),
                                      Conv3d(96,192, kernel_size=3, padding=2),
                                      nn.BatchNorm3d(192),
                                      nn.ReLU(),
                                      MaxPool3d(kernel_size=3, stride=(2,2,1)))

        else:
            self.stem = nn.Sequential(Conv3d(1, 48, kernel_size=3, stride=(2, 2, 2), bias=True, padding=2),
                                      nn.BatchNorm3d(48),
                                      nn.ReLU(),
                                      MaxPool3d(kernel_size=3, stride=(2, 2, 2)),
                                      Conv3d(48, 96, kernel_size=3, bias=True, padding=2),
                                      nn.BatchNorm3d(96),
                                      nn.ReLU(),
                                      Conv3d(96, 192, kernel_size=3, padding=2),
                                      nn.BatchNorm3d(192),
                                      nn.ReLU(),
                                      MaxPool3d(kernel_size=3, stride=(2, 2, 2)))


        self.Block1 = nn.Sequential(InceptionBlock(192),
                                    InceptionBlock(480),
                                    FireModule(480))
        self.Block2 = nn.Sequential(InceptionBlock(512),
                                    InceptionBlock(480),
                                    FireModule(480))
        self.Block3 = nn.Sequential(InceptionBlock(512),
                                    InceptionBlock(480),
                                    FireModule(480))
        # self.Block4 = nn.Sequential(InceptionBlock(512),
        #                             InceptionBlock(480),
        #                             FireModule(480))

        self.average_pool = AvgPool3d(kernel_size=2)

        self.dense512 = Linear(512, 256)
        self.dense256 = Linear(256, 128)
        self.dense128 = Linear(128, 64)

        self.regressor = Linear(64, 1)

        self.relu = ReLU()


    def forward(self, x):


        stem = self.stem(x)

        b1 = self.Block1(stem)
        b2 = self.Block2(b1)
        b3 = self.Block3(b2)
        # b4 = self.Block4(b3).view(-1, 512)

        avgpool = self.average_pool(b3)
        avgpool = avgpool.view(-1, 512)

        d1 = self.relu(self.dense512(avgpool))
        d2 = self.relu(self.dense256(d1))
        d3 = self.relu(self.dense128(d2))

        output = self.regressor(d3)


        return output


class CNN3D:
    def __init__(self, chunk=True):

        self.CNN = CNNModel(chunk=chunk)
        self.loss = MSELoss()
        self.chunk = chunk

    def _validation_eval(self, val_data):
        val_mean_loss = []
        self.CNN.eval()
        with torch.no_grad():
            for t, (x, y) in enumerate(val_data):
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)

                y_pred = self.CNN(x)
                loss = self.loss(y_pred, y)
                val_mean_loss.append(loss)

        return torch.mean(torch.tensor(val_mean_loss)).item()

    def _test_eval(self, test_data):
        val_mean_loss = []
        self.CNN.eval()
        with torch.no_grad():
            for t, (x, y) in enumerate(test_data):
                if self.chunk:
                    x = x.to(device=device, dtype=torch.float32)
                    y = y.to(device=device, dtype=torch.float32)
                    chunks_pred = []
                    for c in range(0, x.shape[1]):
                        chunk = x[:, c]
                        y_chunk = self.CNN(chunk)
                        chunks_pred.append(y_chunk.detach.numpy())
                    y_pred = torch.tensor(np.mean(chunks_pred)).to(device)
                    loss = self.loss(y_pred, y)
                    val_mean_loss.append(loss)
                else:
                    x = x.to(device=device, dtype=torch.float32)
                    y = y.to(device=device, dtype=torch.float32)

                    y_pred = self.CNN(x)
                    loss = self.loss(y_pred, y)
                    val_mean_loss.append(loss)

        return torch.mean(torch.tensor(val_mean_loss)).item()

    def train(self, train_data, val_data, test_data, epochs=100, learning_rate=0.00001, momentum=0.9,
              print_every=10, save_every=10):

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.optimizer = optim.SGD(self.CNN.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, verbose=True, threshold_mode="abs",patience=5)
        self.CNN.to(device)

        for e in range(epochs):
            mean_epoch_train_loss = []
            for t, (x, y) in tqdm(enumerate(train_data), desc="EPOCH " + str(e),total=len(train_data)):

                self.CNN.train()

                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)

                y_pred = self.CNN(x)
                loss = self.loss(y_pred, y)
                mean_epoch_train_loss.append(loss.item())

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                if t % print_every == 0:
                    print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))

                if t % save_every == 0:
                    if self.chunk:
                        torch.save(self.CNN.state_dict(), "models/CNN_chunk_latest.pt")
                    else:
                        torch.save(self.CNN.state_dict(), "models/CNN_latest.pt")

            self.train_losses.append(np.mean(mean_epoch_train_loss))
            val_loss = self._validation_eval(val_data)
            scheduler.step(val_loss)
            self.val_losses.append(val_loss)
            print('Epoch: %d, Validation Loss = %.4f' % (e, val_loss))
            test_loss = self._test_eval(test_data)
            self.test_losses.append(test_loss)
            print('Epoch: %d, Testing Loss = %.4f' % (e, test_loss))
            self.plot_loss()

    def plot_loss(self):

        plt.plot(self.train_losses, label="training loss")
        plt.plot(self.test_losses, label="testing loss")
        plt.plot(self.val_losses, label="validation loss")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss as a function of the training epochs for training, validation and testing data")
        plt.savefig("graphics/lossovertime.png")
        plt.close()

        if self.chunk:
            np.save("tmp/train_loss_chunk.npy", np.array(self.train_losses))
            np.save("tmp/test_loss_chunk.npy", np.array(self.test_losses))
            np.save("tmp/val_loss_chunk.npy", np.array(self.val_losses))
        else:
            np.save("tmp/train_loss.npy", np.array(self.train_losses))
            np.save("tmp/test_loss.npy", np.array(self.test_losses))
            np.save("tmp/val_loss.npy", np.array(self.val_losses))


    def load_model(self):
        if self.chunk:
            self.CNN.load_state_dict(torch.load("models/CNN_chunk_latest.pt"))
            self.train_losses = np.load("tmp/train_loss_chunk.npy").tolist()
            self.val_losses = np.load("tmp/val_loss_chunk.npy").tolist()
            self.test_losses = np.load("tmp/val_loss_chunk.npy").tolist()
        else:
            self.CNN.load_state_dict(torch.load("models/CNN_latest.pt"))
            self.train_losses = np.load("tmp/train_loss.npy").tolist()
            self.val_losses = np.load("tmp/val_loss.npy").tolist()
            self.test_losses = np.load("tmp/val_loss.npy").tolist()


    def infer(self, test_data):
        self.CNN.eval()  # set model to evaluation mode

        preds = []

        with torch.no_grad():
            for x, y in test_data:
                if self.chunk:
                    x = x.to(device=device, dtype=torch.float32)  # move to device
                    chunks_pred = []
                    for c in range(0, x.shape[1]):
                        chunk = x[:,c]
                        y_preds = self.CNN(chunk)
                        chunks_pred.append(y_preds.detach().numpy())
                    preds.append(np.mean(chunks_pred))
                else:
                    x = x.to(device=device, dtype=torch.float32)  # move to device
                    y_preds = self.CNN(x)
                    preds.append(y_preds.detach().numpy())

        return np.array(preds)
