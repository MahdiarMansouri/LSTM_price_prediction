import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

def load_and_normalize_data(filepath, last_n_days=30):
    df = pd.read_csv(filepath)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    close_prices = df[['Close']].values
    # Normalize the data
    normalized_data = scaler.fit_transform(close_prices)
    # Exclude the last 30 days for the training set
    training_data = normalized_data[:-last_n_days]
    return torch.FloatTensor(training_data), torch.FloatTensor(normalized_data), scaler, df


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw - 1):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# class LSTMModel(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size
#         self.lstm = nn.LSTM(input_size, hidden_layer_size)
#         self.linear = nn.Linear(hidden_layer_size, output_size)
#         self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
#                             torch.zeros(1, 1, self.hidden_layer_size))
#
#     def forward(self, input_seq):
#         lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
#         predictions = self.linear(lstm_out.view(len(input_seq), -1))
#         return predictions[-1]


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=5, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)  # Ensure batch_first is True
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # No need for initial hidden state, let LSTM handle it internally
        lstm_out, _ = self.lstm(input_seq)  # Use input_seq directly
        # lstm_out, _ = self.lstm(lstm_out)  # Use input_seq directly
        # Take the output for the last time step for each sequence in the batch
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


class LSTM_GRU_Model(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # LSTM Layer
        self.init_lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.lstm = nn.LSTM(hidden_layer_size, hidden_layer_size, batch_first=True)

        # GRU Layer - input size matches LSTM output size (hidden_layer_size)
        self.gru = nn.GRU(hidden_layer_size, hidden_layer_size // 2, batch_first=True)
        self.s_gru = nn.GRU(hidden_layer_size // 2, hidden_layer_size //4, batch_first=True)
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
        gru_out, _ = self.s_gru(gru_out)
        gru_out, _ = self.s2_gru(gru_out)
        # gru_out, _ = self.gru(gru_out)
        # gru_out, _ = self.gru(gru_out)


        # We use the output of the GRU layer for our predictions
        predictions = self.linear(gru_out[:, -1, :])
        return predictions


def train_model(model, train_loader, loss_function, optimizer, epochs=10):
    for i in range(epochs):
        s0 = datetime.now()
        for seq, labels in train_loader:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            # print(seq.shape)

            y_pred = model(seq)
            labels = labels.view(-1, 1)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        # if i % 2 == 0:
        timing = datetime.now() - s0
        print(f'Epoch {i} | loss: {single_loss.item()} | time: {timing}')


def evaluate_model(model, test_data, scaler):
    model.eval()
    test_predictions = []
    actual_prices = []
    with torch.no_grad():
        for seq, labels in test_data:
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_predictions.append(model(seq).item())
            actual_prices.append(labels.item())

    predicted_prices_scaled = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
    actual_prices_scaled = scaler.inverse_transform(np.array(actual_prices).reshape(-1, 1))

    # Calculate Mean Squared Error
    rmse = np.sqrt(np.mean((predicted_prices_scaled - actual_prices_scaled) ** 2))
    print(f'Root Mean Squared Error: {rmse}')
    return rmse


def rolling_window_evaluation(model, data, scaler, seq_length, last_n_days=30):
    # Extract the last N days for the rolling window prediction
    last_n_index = -last_n_days
    train_data = data[:last_n_index]
    test_data = data[last_n_index - seq_length:]  # Include additional data for initial sequence

    predictions = []
    actuals = test_data[seq_length:]  # Actuals to compare against predictions

    model.eval()
    with torch.no_grad():
        for i in range(len(actuals)):
            seq = test_data[i:i + seq_length]  # Get current sequence
            seq = seq.view(1, seq_length, 1)  # Reshape for model input

            # Predict
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            prediction = model(seq)

            # Store prediction
            predictions.append(prediction.item())

            # Update test_data with actual value for rolling window
            if i < len(actuals) - 1:  # Prevent out-of-bounds error
                test_data[seq_length + i] = actuals[i]

    # Scale predictions and actuals back to original scale
    predictions_scaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals_scaled = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

    # Calculate Mean Squared Error
    rmse = np.sqrt(np.mean((predictions_scaled - actuals_scaled) ** 2))
    print(f'Root Mean Squared Error over last {last_n_days} days: {rmse}')

    return predictions_scaled, actuals_scaled


def plot_predictions(df, predictions_scaled, last_n_days=30):
    # Plot the full dataset
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Actual Prices', color='grey', alpha=0.5)

    # Highlight the last N days in the dataset for actual prices
    plt.plot(df.index[-last_n_days:], df['Close'][-last_n_days:], label='Last N Days Actual Prices', color='blue')

    # Assuming predictions_scaled is a numpy array with the shape (last_n_days, 1)
    # And the last N days are the target of these predictions
    # Create a range for the last N days
    prediction_dates = df.index[-last_n_days:]
    plt.plot(prediction_dates, predictions_scaled, label='Predicted Prices', linestyle='--', color='red')

    plt.title('Bitcoin Price Prediction - Full Year with Last N Days Highlighted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    filepath = 'BTC-USD.csv'
    seq_length = 20

    # Load and normalize data, excluding the last 30 days from the training data
    training_data, full_data, scaler, df = load_and_normalize_data(filepath)

    # Create inout sequences for the training data
    train_sequences = create_inout_sequences(training_data, seq_length)
    train_loader = DataLoader(train_sequences, batch_size=64, shuffle=True)
    #
    # for seq, labels in train_loader:
    #     print(seq.shape)  # Should be [64, seq_length, 1] or similar
    #     print(labels.shape)  # Should match the expected output shape
    #     break  # Just to check the first batch

    # Instantiate the model, define loss and optimizer
    model = LSTM_GRU_Model(hidden_layer_size=3200)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('device is: ', device)
    #
    # # Move your model to the GPU
    # model.to(device)
    #
    # # Move your data to the GPU
    # train_loader = train_loader.to(device)
    # train_sequences = train_sequences.to(device)

    # Train the model
    train_model(model, train_loader, loss_function, optimizer, epochs=100)


    # Evaluate the model using the rolling window approach on the last 30 days
    # Note: 'full_data' contains the normalized data including the last 30 days
    # After calling rolling_window_evaluation in the main workflow:
    predictions_scaled, actuals_scaled = rolling_window_evaluation(model, full_data[-(30 + seq_length):], scaler,
                                                                   seq_length)

    # Now plot the predictions against the actual prices
    # Example usage
    # Make sure predictions_scaled is prepared correctly
    # predictions_scaled = scaler.inverse_transform(predictions.reshape(-1, 1))

    plot_predictions(df, predictions_scaled, last_n_days=30)


