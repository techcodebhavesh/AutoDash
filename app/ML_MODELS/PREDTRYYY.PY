import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Load the dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Preprocess the data
def preprocess_data(data, feature_columns, target_column):
    data = data.dropna(subset=[target_column])
    X = data[feature_columns].values
    y = data[target_column].values
    
    # Scaling features
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_X, scaler_y

# Prepare data for LSTM
def prepare_lstm_data(X, y, window_size=10):
    X_lstm, y_lstm = [], []
    for i in range(len(X) - window_size):
        X_lstm.append(X[i:i + window_size])
        y_lstm.append(y[i + window_size])
    return np.array(X_lstm), np.array(y_lstm)

# Define and train the LSTM model
def train_lstm_model(X_train, y_train, window_size):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, X_train.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

def predict_future_values(model, X, window_size, scaler_y, n_predictions=20):
    # Transform the last window of data
    X_transformed = X[-window_size:]
    future_predictions = []
    
    # Loop to generate future predictions
    for _ in range(n_predictions):
        # Reshape last_window to match the model input shape
        next_prediction = model.predict(X_transformed.reshape(1, window_size, X_transformed.shape[1]))
        next_prediction = next_prediction[0, 0]  # Adjust indexing if needed
        
        # Append prediction and update the last_window
        future_predictions.append(next_prediction)
        X_transformed = np.roll(X_transformed, shift=-1, axis=0)
        X_transformed[-1] = np.array([next_prediction] * X_transformed.shape[1])
    
    # Inverse transform the predictions
    future_predictions = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    return future_predictions

# Plot results
def plot_results(original_series, predictions, n_recent=10):
    recent_actual = original_series[-n_recent:]
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(original_series) - n_recent, len(original_series)), recent_actual, label='Recent Actual Data', color='blue')
    plt.plot(range(len(original_series), len(original_series) + len(predictions)), predictions, label='Predicted Data', color='red', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Prediction vs Recent Actual Data')
    plt.legend()
    plt.show()

def main(file_path, feature_columns, target_column):
    data = load_data(file_path)
    if data is not None:
        X, y, scaler_X, scaler_y = preprocess_data(data, feature_columns, target_column)
        X_lstm, y_lstm = prepare_lstm_data(X, y, window_size=10)
        X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=0)
        model = train_lstm_model(X_train, y_train, window_size=10)
        y_pred = model.predict(X_test).flatten()
        mse = mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), scaler_y.inverse_transform(y_pred.reshape(-1, 1)))
        print(f"Mean Squared Error: {mse:.4f}")
        future_predictions = predict_future_values(model, X, window_size=10, scaler_y=scaler_y)
        print(f"Future Predictions: {future_predictions}")
        plot_results(data[target_column].values, future_predictions)

file_path = 'D:/AutoDash/DailyDelhiClimateTrain.csv'
feature_columns = ['meantemp']
target_column = 'meanpressure'
main(file_path, feature_columns, target_column)
