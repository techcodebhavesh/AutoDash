import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from app.ML_MODELS.Files import NGINX_FOLDER,NGINX_URL,getFile
import os

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
# Train the ARIMA model
def train_model(y, order):
    model = ARIMA(y, order=order)
    model_fit = model.fit()
    return model_fit
# Evaluate the model
def evaluate_model(model_fit, y, train_size,image_url):
    # Split the data
    train = y[:train_size]
    test = y[train_size:]
    
    # Make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
    
    # Calculate error metrics
    mse = mean_squared_error(test, predictions)
    print(f'Mean Squared Error: {mse:.4f}')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y.index, y, label='Original')
    plt.plot(test.index, predictions, label='Predicted', color='red')
    plt.legend()
    plt.title('ARIMA Forecasting')
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
    
    # Train the ARIMA model (order=(p,d,q) where p=AR terms, d=differencing, q=MA terms)
    order = (5, 1, 0)  # Example order, you might need to adjust based on your data
    model_fit = train_model(y, order)
    
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
    date_column = params.get("date_column")
    target_column = params.get("target_column")
    image=getFile()
    imageFolder=os.path.join(NGINX_FOLDER,"image2.png")
    imageUrl=NGINX_URL+image

    df = load_data(csv_path, date_column, target_column)

    # Preprocess the data
    y = preprocess_data(df, target_column)

    # Split the data into train and test sets
    train_size = int(len(y) * 0.8)

    # Train the ARIMA model (order=(p,d,q) where p=AR terms, d=differencing, q=MA terms)
    order = (5, 1, 0)  # Example order, you might need to adjust based on your data
    model_fit = train_model(y, order)

    # Evaluate the model
    evaluate_model(model_fit, y, train_size,imageFolder)

    # Forecast future values
    steps = 10  # Number of future steps to forecast
    forecast = forecast_future(model_fit, steps)
    print("Forecasted values:")
    print(forecast)
    return {"forecast":forecast,"image":image,"url":imageUrl}
