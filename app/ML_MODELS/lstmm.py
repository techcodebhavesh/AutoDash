import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from app.ML_MODELS.Files import NGINX_FOLDER,NGINX_URL,getFile

# Load the time series data
def load_data(csv_path, date_column, target_column):
    df = pd.read_csv(csv_path, parse_dates=[date_column], index_col=date_column)
    return df

# Preprocess the data
def preprocess_data(df, target_column):
    df = df.sort_index()
    df = df.fillna(method='ffill')
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[target_column]])
    
    return scaled_data, scaler
# Create sequences for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)
# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the LSTM model
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return history
# Evaluate the model
def evaluate_model(model, X, y, scaler, sequence_length,image_url):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    y = scaler.inverse_transform(y)
    
    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error: {mse:.4f}')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y)), y, label='Original')
    plt.plot(range(len(predictions)), predictions, label='Predicted', color='red')
    plt.legend()
    plt.title('LSTM Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
    plt.savefig(image_url)
# Forecast future values
def forecast_future(model, last_sequence, sequence_length, scaler, steps):
    forecast = []
    current_sequence = last_sequence
    
    for _ in range(steps):
        prediction = model.predict(current_sequence.reshape(1, sequence_length, 1))
        forecast.append(prediction[0,0])
        current_sequence = np.append(current_sequence[1:], prediction, axis=0)
    
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast
# Main function
if __name__ == "__main__":
    # Load your dataset
    csv_path = "ww2.csv"  # Replace with your actual file path
    date_column = "date"   # Replace with your date column name
    target_column = "meantemp"  # Replace with your target column name

    df = load_data(csv_path, date_column, target_column)

    # Preprocess the data
    scaled_data, scaler = preprocess_data(df, target_column)

    # Create sequences
    sequence_length = 10  # Length of the sequences for LSTM
    X, y = create_sequences(scaled_data, sequence_length)

    # Split the data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train the LSTM model
    model = build_lstm_model((X_train.shape[1], 1))
    train_model(model, X_train, y_train, epochs=10, batch_size=32)

    image=NGINX_FOLDER+"image2.png"

    # Evaluate the model
    evaluate_model(model, X_test, y_test, scaler, sequence_length,image)

    # Forecast future values
    last_sequence = scaled_data[-sequence_length:]  # Use the last sequence of the dataset
    steps = 10  # Number of future steps to forecast
    forecast = forecast_future(model, last_sequence, sequence_length, scaler, steps)
    print("Forecasted values:")
    print(forecast)

def run(params):
    # Load your dataset
    csv_path = params.get("csv_path")
    date_column = params.get("date_column")
    target_column = params.get("target_column")
    image=getFile()
    imageFolder=NGINX_FOLDER+image
    imageUrl=NGINX_URL+image


    df = load_data(csv_path, date_column, target_column)

    # Preprocess the data
    scaled_data, scaler = preprocess_data(df, target_column)

    # Create sequences
    sequence_length = 10  # Length of the sequences for LSTM
    X, y = create_sequences(scaled_data, sequence_length)

    # Split the data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train the LSTM model
    model = build_lstm_model((X_train.shape[1], 1))
    train_model(model, X_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, scaler, sequence_length,imageFolder)

    # Forecast future values
    last_sequence = scaled_data[-sequence_length:]  # Use the last sequence of the dataset
    steps = 10  # Number of future steps to forecast
    forecast = forecast_future(model, last_sequence, sequence_length, scaler, steps)
    print("Forecasted values:")
    print(forecast)
    return {"forecast":forecast,"image":image,"url":imageUrl}
