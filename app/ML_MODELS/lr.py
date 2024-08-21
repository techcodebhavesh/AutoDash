import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the CSV data
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Preprocess the data
def preprocess_data(df, target_column, feature_column):
    # Handle missing or non-numeric values
    df = df.dropna(subset=[feature_column, target_column])
    df[feature_column] = pd.to_numeric(df[feature_column], errors='coerce')  # Convert to numeric, set non-convertibles to NaN
    df = df.dropna(subset=[feature_column])  # Drop rows where conversion failed

    # Select only the chosen feature column plus the target column
    df = df[[feature_column, target_column]]

    # Separate features and target
    X = df[[feature_column]]
    y = df[target_column]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_column

# Train the Linear Regression model
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
def predict_for_input(model, scaler, feature_column,user_input):
    # Collect feature value from the user
    # value = float(input(f"Enter value for {feature_column}: "))

    # Convert to NumPy array and scale the input
    # user_input = np.array([[value]])
    user_input_scaled = scaler.transform(user_input)

    # Make prediction
    prediction = model.predict(user_input_scaled)
    return prediction

# Main function
if __name__ == "__main__":
    # Load your dataset
    csv_path = "D:/AutoDash/app/ML_MODELS/w.csv"
    target_column = "Price"  # Replace with your target column name
    df = load_data(csv_path)

    # Prompt user to select feature column
    print("Available feature columns:")
    for col in df.columns:
        if col != target_column:
            print(f"- {col}")

    feature_column = input("Enter the feature column you want to use for prediction: ").strip()

    # Preprocess the data based on selected feature column
    X, y, scaler, feature_column = preprocess_data(df, target_column, feature_column)

    # Train the model
    model, X_test, y_test = train_model(X, y)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Predict target value for user input
    prediction = predict_for_input(model, scaler, feature_column)
    print(f"Predicted target value: {prediction[0]}")

def run(params):
# Load your dataset
    csv_path = params.get('csv_path')
    target_column = params.get('target_col')  # Replace with your target column name
    df = load_data(csv_path)
    user_input = params.get("user_input")
    user_input = np.array(user_input).reshape(-1, 1)

    # # Prompt user to select feature column
    # print("Available feature columns:")
    # for col in df.columns:
    #     if col != target_column:
    #         print(f"- {col}")

    # feature_column = input("Enter the feature column you want to use for prediction: ").strip()
    feature_column=params.get('feature_col')
    # feature_column="Inches"
    # Preprocess the data based on selected feature column
    X, y, scaler, feature_column = preprocess_data(df, target_column, feature_column)

    # Train the model
    model, X_test, y_test = train_model(X, y)

    # Evaluate the model
    mes,r2=evaluate_model(model, X_test, y_test)

    # Predict target value for user input
    prediction = predict_for_input(model, scaler, feature_column,user_input)
    print(f"Predicted target value: {prediction[0]}")   
    return prediction[0],mes,r2
