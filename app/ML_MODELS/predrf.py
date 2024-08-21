import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load CSV data
df = pd.read_csv('D:/AutoDash/w.csv')

# Define features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Preprocessing pipelines
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define base models
rf = RandomForestRegressor()
gbr = GradientBoostingRegressor()
svr = SVR()

# Combine models in a voting regressor
voting_regressor = VotingRegressor([('rf', rf), ('gbr', gbr), ('svr', svr)])

# Create a pipeline with preprocessing and modeling
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', voting_regressor)])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predict the next 20 values
y_pred = pipeline.predict(X_test)
predictions = y_pred[:20]

# Plot historical and predicted values
plt.figure(figsize=(14, 7))

# Plot the historical values
plt.plot(np.arange(len(y_test)), y_test, label='Historical Data', color='blue', linestyle='--')

# Plot the predicted values
plt.plot(np.arange(len(y_test), len(y_test) + 20), predictions, label='Predicted Data', color='red')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Historical and Predicted Values')
plt.legend()
plt.grid(True)
plt.show()
