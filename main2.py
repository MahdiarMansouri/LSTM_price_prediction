import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def prepare_datasets(data, features, last_n_days, batch_size, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                     use_scaler=True):
    # Ensure the split ratios sum to 1
    # assert (train_ratio + val_ratio + test_ratio) == 1, "Ratios must sum to 1"

    # Select the specified features from the dataset
    selected_data = data[features].values  # Assuming 'data' is a DataFrame

    # Normalize the dataset if use_scaler is True
    if use_scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(selected_data)
    else:
        scaled_data = selected_data

    # Create sequences
    X, y = [], []
    for i in range(last_n_days, len(scaled_data)):
        X.append(scaled_data[i - last_n_days:i])
        y.append(scaled_data[i, :])  # Assumes you want to predict all features at the next time step
    X, y = np.array(X), np.array(y)

    # Convert to PyTorch tensors
    X_tensor, y_tensor = torch.FloatTensor(X), torch.FloatTensor(y)

    # Calculate split sizes
    total_samples = X_tensor.size(0)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    # Split the data
    train_X, train_y = X_tensor[:train_size], y_tensor[:train_size]
    val_X, val_y = X_tensor[train_size:train_size + val_size], y_tensor[train_size:train_size + val_size]
    test_X, test_y = X_tensor[-test_size:], y_tensor[-test_size:]

    # Wrap the datasets with TensorDataset
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    test_dataset = TensorDataset(test_X, test_y)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class LSTM_GRU_Model(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # LSTM Layer
        self.init_lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.lstm = nn.LSTM(hidden_layer_size, hidden_layer_size, batch_first=True)

        # GRU Layer - input size matches LSTM output size (hidden_layer_size)
        self.gru = nn.GRU(hidden_layer_size, hidden_layer_size , batch_first=True)
        self.s_gru = nn.GRU(hidden_layer_size // 2, hidden_layer_size, batch_first=True)
        self.s2_gru = nn.GRU(hidden_layer_size // 4, hidden_layer_size, batch_first=True)
        self.gru_after_bi = nn.GRU(hidden_layer_size * 2, hidden_layer_size, batch_first=True)

        # Linear layer to get to the output size
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # LSTM layer
        lstm_out, _ = self.init_lstm(input_seq)
        # lstm_out, _ = self.lstm(lstm_out)
        # lstm_out, _ = self.lstm(lstm_out)
        # lstm_out, _ = self.lstm(lstm_out)

        # GRU layer
        gru_out, _ = self.gru(lstm_out)
        # lstm_out, _ = self.lstm(gru_out)
        # gru_out, _ = self.s_gru(gru_out)
        # gru_out, _ = self.s2_gru(gru_out)
        # gru_out, _ = self.gru(gru_out)
        # gru_out, _ = self.gru(gru_out)

        # We use the output of the GRU layer for our predictions
        predictions = self.linear(gru_out[:, -1, :])
        return predictions


def train_val_model(model, train_loader, val_loader, loss_function, optimizer, device, epochs=10):
    model.to(device)

    # Initialize loss lists
    train_loss_list = []
    val_loss_list = []

    for i in range(epochs):
        model.train()
        train_running_loss = 0.0
        s0 = datetime.now()
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)

            optimizer.zero_grad()
            y_pred = model(seq)
            labels = labels.view(-1, 1)

            single_loss = loss_function(y_pred, labels)
            train_running_loss += single_loss.item() * seq.size(0)

            single_loss.backward()
            optimizer.step()

        # Calculate train loss for this epoch
        train_loss = train_running_loss / len(train_loader.dataset)
        train_loss_list.append(train_loss)

        # Validation loop
        val_running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.to(device), labels.to(device)

                predict_price = model(seq)
                labels = labels.view(-1, 1)

                single_loss = loss_function(predict_price, labels)
                val_running_loss += single_loss.item() * seq.size(0)

        # Calculate validation loss for this epoch
        val_loss = val_running_loss / len(val_loader.dataset)
        val_loss_list.append(val_loss)

        timing = datetime.now() - s0
        print(f'Epoch {i} | train_loss: {train_loss:.8f} | val_loss: {val_loss:.8f} | time: {timing}')

    return train_loss_list, val_loss_list


def evaluate_model(model, test_loader, device, loss_function):
    model.eval()
    test_running_loss = 0.0

    with torch.no_grad():
        for seq, labels in test_loader:
            seq, labels = seq.to(device), labels.to(device)
            labels = labels.view(-1, 1)

            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            test_running_loss += single_loss.item() * seq.size(0)

    test_loss = test_running_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.8f}')

    return test_loss


def scatter_plot_predictions(actual, predicted, title='Actual vs Predicted'):
    plt.figure(figsize=(10, 6))
    plt.scatter([a for a in range(len(actual))], actual, label='Actual Values', color='blue')
    plt.scatter([a for a in range(len(predicted))], predicted, label='Predicted Values', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(actual, predicted, title='Actual vs Predicted'):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Values', color='blue')
    plt.plot(predicted, label='Predicted Values', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_losses(train_losses, val_losses, title='Training and Validation Loss'):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Define dataset and params
    features_to_use = ['close']
    df = pd.read_csv('data/XAUUSD-M15')
    seq_length = 672
    batch_size = 256
    scaler = True

    # Creating dataloaders
    train, val, test = prepare_datasets(df, features_to_use, last_n_days=seq_length, batch_size=batch_size,
                                        use_scaler=scaler)
    print(
        f'DataLoaders are Created with "Sequence Length = {seq_length}" and "Batch Size = {batch_size}" and "Scaler = {scaler}":')
    print(f'Train Length: {len(train)}')
    print(f'Val Length: {len(val)}')
    print(f'Test Length: {len(test)}')
    print('-' * 50)

    # define model and loss_func and optimizer and device
    model = LSTM_GRU_Model(hidden_layer_size=256)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device is: ', device)
    print('-' * 50)

    # Train and Evaluate model
    epochs = 200
    print(f'Training model in {epochs} epochs: ')
    train_losses, val_losses = train_val_model(model, train, val, loss_function, optimizer, device, epochs=epochs)
    evaluate_model(model, test, device, loss_function)
    print('-' * 50)

    # Make predictions and plot data
    actual_prices = []
    predicted_prices = []

    model.eval()
    for dataloader in [train, val, test]:
        with torch.no_grad():
            for seq, labels in dataloader:
                seq, labels = seq.to(device), labels.to(device)

                # Predict
                predictions = model(seq)

                # Collect actual and predicted values
                actual_prices.extend(labels.cpu().view(-1).numpy())
                predicted_prices.extend(predictions.cpu().view(-1).numpy())

    # plot results
    title = 'XAUUSD-M15 | sl=672 | bs=256 | hl=256'
    scatter_plot_predictions(actual_prices, predicted_prices, title=title)
    plot_predictions(actual_prices, predicted_prices, title=title)
    plot_losses(train_losses, val_losses)
