import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt

# Load your data into a DataFrame 'df'
df = pd.read_csv('D:/AutoDash/DailyDelhiClimateTrain.csv')

# Function to automatically detect and convert date format
def convert_date_format(df, date_column):
    date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
    
    for date_format in date_formats:
        try:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
            print(f"Successfully converted dates using format: {date_format}")
            return df
        except ValueError:
            continue

    # If all formats fail, try automatic conversion
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    if df[date_column].isnull().any():
        raise ValueError(f"Could not convert some dates in column {date_column}")

# Convert 'date' column and set it as the index
df = convert_date_format(df, 'date')
df.set_index('date', inplace=True)

# Step 1: Detect the type of data
def detect_data_type(df):
    if isinstance(df.index, pd.DatetimeIndex):
        return 'time_series'
    else:
        return 'tabular'

# Step 2: Check stationarity for time series data
def check_stationarity(data):
    result = adfuller(data)
    p_value = result[1]
    return p_value < 0.05  # If p-value is less than 0.05, data is stationary

# Step 3: Prepare data for LSTM
def prepare_lstm_data(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# Step 4: Preprocess the data
def preprocess_data(df, target_column):
    # Handle missing values
    df = df.dropna()

    # Handle categorical data
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y, label_encoders

# Step 5: Choose the appropriate model
def choose_model(df, target_column):
    data_type = detect_data_type(df)

    if data_type == 'time_series':
        if check_stationarity(df[target_column].values):
            return 'ARIMA'  # Choose ARIMA for stationary time series
        else:
            return 'LSTM'  # Choose LSTM for non-stationary time series
    else:
        X, y, _ = preprocess_data(df, target_column)
        
        if X.shape[1] > 10:  # Assume more than 10 features as complex data
            return 'XGBoost'  # Choose XGBoost for complex tabular data
        else:
            return 'LinearRegression'  # Choose Linear Regression for simpler data

# Step 6: Implement the selected model
def implement_model(model_name, df, target_column):
    try:
        if model_name == 'ARIMA':
            # Implement ARIMA model
            data = df[target_column].values
            model = ARIMA(data, order=(5, 1, 0))  # Order can be tuned
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=20)  # Predict next 20 rows
            print(f"ARIMA Predictions: {predictions}")
            plot_predictions(df[target_column], predictions, model_name)
        
        elif model_name == 'LSTM':
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[target_column].values.reshape(-1, 1))

            # Prepare data for LSTM
            look_back = 10  # Number of previous time steps to consider
            X, y = prepare_lstm_data(scaled_data, look_back)

            # Reshape X to be [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            # LSTM Model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X, y, epochs=100, batch_size=32, verbose=2)

            # Predict the next 20 rows
            last_known_data = scaled_data[-look_back:]
            predictions = []
            for _ in range(20):
                next_pred = model.predict(np.reshape(last_known_data, (1, look_back, 1)))
                predictions.append(next_pred[0, 0])
                last_known_data = np.append(last_known_data[1:], next_pred, axis=0)

            # Inverse scale the predictions
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            print(f"LSTM Predictions: {predictions}")
            plot_predictions(df[target_column], predictions, model_name)

        elif model_name == 'XGBoost':
            X, y, _ = preprocess_data(df, target_column)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = XGBRegressor()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            print(f"XGBoost MSE: {mse}")
            print(f"XGBoost Predictions: {predictions[:20]}")
            plot_predictions(y_test, predictions[:20], model_name)

        elif model_name == 'LinearRegression':
            X, y, _ = preprocess_data(df, target_column)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            print(f"Linear Regression MSE: {mse}")
            print(f"Linear Regression Predictions: {predictions[:20]}")
            plot_predictions(y_test, predictions[:20], model_name)

    except NotFittedError as e:
        print(f"Model could not be fitted: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Step 7: Plot the results
def plot_predictions(true_data, predictions, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(true_data.index[-len(predictions):], true_data[-len(predictions):], label='Actual Data')
    plt.plot(true_data.index[-len(predictions):], predictions, label='Predictions', color='red')
    plt.title(f'{model_name} Model Predictions')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    # plt.show()

# Step 8: Run the model selection and implementation
target_column = 'meantemp'  # Replace with the actual target column name
model_name = choose_model(df, target_column)
print(f"Selected Model: {model_name}")
implement_model(model_name, df, target_column)
