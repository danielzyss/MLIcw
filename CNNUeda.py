from parameters import *


class CNNModelUeda(nn.Module):

    def __init__(self, ):
        super(CNNModelUeda, self).__init__()


        self.convnet = nn.Sequential(Conv3d(1, 8, kernel_size=3, stride=1, bias=True, padding=2),
                                  nn.BatchNorm3d(8),
                                  nn.ReLU(),
                                  MaxPool3d(kernel_size=2, stride=2),
                                  Conv3d(8, 16, kernel_size=3, stride=1, bias=True, padding=2),
                                  nn.BatchNorm3d(16),
                                  nn.ReLU(),
                                  MaxPool3d(kernel_size=2, stride=2),
                                  Conv3d(16, 32, kernel_size=3, stride=1, bias=True, padding=2),
                                  nn.BatchNorm3d(32),
                                  nn.ReLU(),
                                  MaxPool3d(kernel_size=2, stride=2),
                                  Conv3d(32, 64, kernel_size=3, stride=1, bias=True, padding=2),
                                  nn.BatchNorm3d(64),
                                  nn.ReLU(),
                                  MaxPool3d(kernel_size=2, stride=2))

        self.dense1 = nn.Sequential(nn.Linear(25088, 9600), nn.ReLU())
        self.dense2 = nn.Sequential(nn.Linear(9600, 1024), nn.ReLU())
        self.dense3 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU())


        self.regressor = Linear(1024, 1)


    def forward(self, x):


        cnet = self.convnet(x)
        d1 = self.dense1(cnet.view(-1, 25088))
        d2 = self.dense2(d1)
        d3 = self.dense3(d2)
        output = self.regressor(d3)

        return output


class CNN3Dueda:
    def __init__(self):

        self.CNN = CNNModelUeda()
        self.loss = MSELoss()

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


    def train(self, train_data, val_data, test_data, epochs=100, learning_rate=0.00005, momentum=0.9,
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
                    torch.save(self.CNN.state_dict(), "models/CNN_ueda_latest.pt")

            mean_train_loss = np.mean(mean_epoch_train_loss)
            self.train_losses.append(mean_train_loss)
            print('Epoch: %d, Training Loss = %.4f' % (e, mean_train_loss))

            val_loss = self._validation_eval(val_data)
            scheduler.step(val_loss)
            self.val_losses.append(val_loss)
            print('Epoch: %d, Validation Loss = %.4f' % (e, val_loss))

            test_loss = self._validation_eval(test_data)
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
        plt.savefig("graphics/lossovertime_ueda.png")
        plt.close()

        np.save("tmp/train_ueda_loss.npy", np.array(self.train_losses))
        np.save("tmp/test_ueda_loss.npy", np.array(self.test_losses))
        np.save("tmp/val_ueda_loss.npy", np.array(self.val_losses))


    def load_model(self):
        self.CNN.load_state_dict(torch.load("models/CNN_ueda_latest.pt"))
        self.train_losses = np.load("tmp/train_ueda_loss.npy").tolist()
        self.val_losses = np.load("tmp/val_ueda_loss.npy").tolist()
        self.test_losses = np.load("tmp/val_ueda_loss.npy").tolist()


    def infer(self, test_data):
        self.CNN.eval()

        preds = []
        ground_truth = []

        with torch.no_grad():
            for x, y in test_data:
                x = x.to(device=device, dtype=torch.float32)
                y_preds = self.CNN(x)
                preds+=y_preds.detach().cpu().numpy().tolist()
                ground_truth += y.cpu().numpy().flatten().tolist()

        return np.array(preds), np.array(ground_truth)
