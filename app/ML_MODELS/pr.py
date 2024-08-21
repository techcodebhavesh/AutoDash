import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV data
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Preprocess the data
def preprocess_data(df, target_column, feature_column, degree=2):
    # Handle missing or non-numeric values
    df = df.dropna(subset=[feature_column, target_column])
    df[feature_column] = pd.to_numeric(df[feature_column], errors='coerce')
    df = df.dropna(subset=[feature_column])  # Drop rows where conversion failed
    
    # Select only the chosen feature column plus the target column
    df = df[[feature_column, target_column]]
    
    # Separate features and target
    X = df[[feature_column]].values
    y = df[target_column].values
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Polynomial feature transformation
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_scaled)
    
    return X_poly, y, scaler, poly, feature_column

# Train the Polynomial Regression model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R-squared: {r2:.4f}')
    
    
    return mse,r2

# Make predictions for user input based on a single feature
def predict_for_input(model, scaler, poly, feature_column,user_input):
    # Collect feature value from the user
    
    
    # Convert to NumPy array, scale the input, and transform polynomial features
    user_input_scaled = scaler.transform(user_input)
    user_input_poly = poly.transform(user_input_scaled)
    
    # Make prediction
    prediction = model.predict(user_input_poly)
    return prediction

# Main function
if __name__ == "__main__":
    # Load your dataset
    csv_path = "D:/AutoDash/w.csv"  # Replace with your actual file path
    target_column = "Price"  # Replace with your target column name
    
    df = load_data(csv_path)
    
    # Prompt user to select feature column
    print("Available feature columns:")
    for col in df.columns:
        if col != target_column:
            print(f"- {col}")
    feature_column = input("Enter the feature column you want to use for prediction: ").strip()
    
    # Preprocess the data based on selected feature column
    X, y, scaler, poly, feature_column = preprocess_data(df, target_column, feature_column, degree=2)
    
    # Train the model
    model, X_test, y_test = train_model(X, y)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Predict target value for user input
    prediction = predict_for_input(model, scaler, poly, feature_column)
    print(f"Predicted target value: {prediction[0]}")


def run(params):
    csv_path = params.get("csv_path")
    target_column = params.get("target_col")
    feature_column = params.get("feature_col")
    user_input = params.get("user_input")
    user_input = np.array(user_input).reshape(-1, 1)

        
    df = load_data(csv_path)
    
    
    # Preprocess the data based on selected feature column
    X, y, scaler, poly, feature_column = preprocess_data(df, target_column, feature_column, degree=2)
    
    # Train the model
    model, X_test, y_test = train_model(X, y)
    
    # Evaluate the model
    mse,r2=evaluate_model(model, X_test, y_test)
    prediction = predict_for_input(model, scaler, poly, feature_column,user_input)
    print(f"Predicted target value: {prediction[0]}")   
    return prediction[0],mse,r2
