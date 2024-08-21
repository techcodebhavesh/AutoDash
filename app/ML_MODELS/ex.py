import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from app.ML_MODELS.Files import NGINX_FOLDER,NGINX_URL,getFile

# Load the time series data
def load_data(csv_path, date_column, target_column):
    df = pd.read_csv(csv_path, parse_dates=[date_column], index_col=date_column)
    return df

# Preprocess the data
def preprocess_data(df, target_column):
    # Ensure the data is sorted by date
    df = df.sort_index()
    
    # Handle missing values by forward-filling
    df = df.fillna(method='ffill')
    
    # Extract target values
    y = df[target_column]
    
    return y
# Train the Exponential Smoothing model
def train_model(y, trend=None, seasonal=None, seasonal_periods=None):
    model = ExponentialSmoothing(y, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    return model_fit
# Evaluate the model
def evaluate_model(model_fit, y, train_size,image_url):
    # Split the data
    train = y[:train_size]
    test = y[train_size:]
    
    # Make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
    
    # Calculate error metrics
    mse = mean_squared_error(test, predictions)
    print(f'Mean Squared Error: {mse:.4f}')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y.index, y, label='Original')
    plt.plot(test.index, predictions, label='Predicted', color='red')
    plt.legend()
    plt.title('Exponential Smoothing Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()
    plt.savefig(image_url)
# Forecast future values
def forecast_future(model_fit, steps):
    forecast = model_fit.forecast(steps=steps)
    return forecast
# Main function
if __name__ == "__main__":
    # Load your dataset
    csv_path = "D:/AutoDash/ww2.csv"  # Replace with your actual file path
    date_column = "date"   # Replace with your date column name
    target_column = "meantemp"  # Replace with your target column name
    
    df = load_data(csv_path, date_column, target_column)
    
    # Preprocess the data
    y = preprocess_data(df, target_column)
    
    # Split the data into train and test sets
    train_size = int(len(y) * 0.8)
    
    # Train the Exponential Smoothing model (adjust parameters based on your data)
    trend = 'add'  # Options: 'add', 'mul', or None
    seasonal = 'add'  # Options: 'add', 'mul', or None
    seasonal_periods = 12  # Set based on your data (e.g., 12 for monthly data with yearly seasonality)
    model_fit = train_model(y, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    
    # Evaluate the model
    evaluate_model(model_fit, y, train_size)
    
    # Forecast future values
    steps = 10  # Number of future steps to forecast
    forecast = forecast_future(model_fit, steps)
    print("Forecasted values:")
    print(forecast)

def run(params):
    # Load your dataset
    csv_path = params.get("csv_path")
    date_column = params.get("date_column");
    target_column = params.get("target_column");
    image=getFile()
    imageFolder=NGINX_FOLDER+image
    imageUrl=NGINX_URL+image

    df = load_data(csv_path, date_column, target_column)
    
    # Preprocess the data
    y = preprocess_data(df, target_column)
    
    # Split the data into train and test sets
    train_size = int(len(y) * 0.8)
    
    # Train the Exponential Smoothing model (adjust parameters based on your data)
    trend = 'add'  # Options: 'add', 'mul', or None
    seasonal = 'add'  # Options: 'add', 'mul', or None
    seasonal_periods = 12  # Set based on your data (e.g., 12 for monthly data with yearly seasonality)
    model_fit = train_model(y, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    
    # Evaluate the model
    evaluate_model(model_fit, y, train_size, imageFolder)
    
    # Forecast future values
    steps = 10  # Number of future steps to forecast
    forecast = forecast_future(model_fit, steps)
    print("Forecasted values:")
    print(forecast)
    return {"forecast":forecast,"image":image,"url":imageUrl}
