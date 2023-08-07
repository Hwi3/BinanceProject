import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from binance.client import Client
import config
from datetime import datetime
from copy import deepcopy as dc
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LSTM(nn.Module):
    def __init__(self, input_size,hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self,x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to("cpu")
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to("cpu")
        out, _ = self.lstm(x, (h0,c0))
        out = self.fc(out[:,-1, :])
        return out

def n_back(df, n_steps):
    df = dc(df)
    df['Opentime'] = pd.to_datetime(df['Opentime'], unit = 'ms')
    df.set_index('Opentime', inplace =True)
    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)
    return df


def main():
    device = torch.device("cpu")
    def train_one_epoch():
        model.train(True)
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 99:
                avg_loss_across_batches = running_loss / 100
                print("Batch {0}, loss: {1:.3f}".format(batch_index + 1, avg_loss_across_batches))

                running_loss = 0.0

        scheduler.step()
        print("lr: ", optimizer.param_groups[0]['lr'])


    def validate_one_epoch():
        model.train(False)
        running_loss = 0.0

        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)

        print('Val Loss {0:.7f}'.format(avg_loss_across_batches))
        print('************************************************')

    now = datetime.now()
    client = Client(config.apiKey, config.apiSecurity)
    currency = 'BTCUSDT'
    interval = '1h'

    # Historical Trade Data #
    trades = client.get_historical_klines(currency,interval,"365 day ago UTC")
    trades_df = pd.DataFrame(trades)
    trades_df.columns = ['Opentime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    trades_df.to_csv(f'./data/{now}.csv')
    trades_df.to_csv(f'./data/recent.csv')
    data = pd.read_csv(f'./data/{now}.csv')
    data = data[['Opentime', 'Close']]

    n = 8
    new_data = n_back(data, n)
    new_data_np = new_data.to_numpy()[:-1]
    X = new_data_np[:, 1:] / 10000
    Y = new_data_np[:, 0] / 10000

    split_index = int(len(X) * 0.95)

    X_train = X[:split_index]
    X_test = X[split_index:]
    Y_train = Y[:split_index]
    Y_test = Y[split_index:]

    X_train = X_train.reshape((-1, n, 1))
    X_test = X_test.reshape((-1, n, 1))
    Y_train = Y_train.reshape((-1, 1))
    Y_test = Y_test.reshape((-1, 1))

    X_train = torch.tensor(X_train).float()
    Y_train = torch.tensor(Y_train).float()
    X_test = torch.tensor(X_test).float()
    Y_test = torch.tensor(Y_test).float()

    train_dataset = TimeSeriesDataset(X_train, Y_train)
    test_dataset = TimeSeriesDataset(X_test, Y_test)

    #######################################

    batch_size = 16
    device = torch.device("cpu")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTM(1, 4, 1)
    model.to(device)

    num_epochs = 50
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer = optimizer, T_0=num_epochs, T_mult=int(num_epochs/10)-1,eta_min=0.00001)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=lambda epoch: 0.95 ** epoch,
                                                  verbose=False)
    print("START TRAINING")
    for epoch in range(num_epochs):
        train_one_epoch()
        validate_one_epoch()

    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()

    plt.plot(Y_train, label='Actual Close')
    plt.plot(predicted, label='Predicted Close')
    plt.savefig("./graph_img.png")
    torch.save(model.state_dict(), 'model.pth')
    plt.show()


if __name__ == "__main__":
    main()

