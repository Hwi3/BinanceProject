import torch
import pandas as pd
import torch.nn as nn
from binance.client import Client
import config
import argparse


client = Client(config.apiKey, config.apiSecurity)

class LSTM(nn.Module):
    def __init__(self, input_size,hidden_size, num_stacked_layers, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size,1)
        self.device = device

    def forward(self,x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0,c0))
        out = self.fc(out[:,-1, :])
        return out

def model_load(device):
    model = LSTM(1, 4, 1, device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    return model

def data_load(currency,interval, n):
    data = client.get_historical_klines(currency, interval, limit=n+1)
    data = pd.DataFrame(data)
    data.columns = ['Opentime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                         'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    data = data.astype({'Close':'float'})
    data = data['Close']
    X = data.to_numpy()
    X = torch.tensor(X[:n]).float()
    return X


def main():

    ##### DEFAULT #####
    n = 8
    interval = '1h'
    currency = 'BTCUSDT'
    device = torch.device("cpu")
    model = model_load(device)
    text = ""


    ############## GET RECENT PRICE ###############
    X = data_load(currency, interval, n)
    cur = X[-1]
    text += f"Previous: {cur}"
    X = X / 10000

    ############# PREDICT NEXT PRICE #############
    y_pred = float(model(X.reshape((1, n, -1)))) * 10000
    text += f"\nPrediction: {y_pred}"
    gap = y_pred - cur
    position = "long" if gap > 0 else "short"
    text += f"\nPosition: {position}"
    print(text)

    return text, position



if __name__ == '__main__':
    main()