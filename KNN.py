import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Generate some example data (replace this with your actual data)
def KNN(X,y,new_data):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data (optional but often recommended for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the KNN regressor
    k_neighbors = min(5, len(X_train_scaled))  # Ensure not to exceed the number of training samples
    knn_regressor = KNeighborsRegressor(n_neighbors=k_neighbors)
    knn_regressor.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = knn_regressor.predict(X_test_scaled)

    # Evaluate the model (for regression, you can use metrics like Mean Squared Error)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Make predictions on new data (replace this with your actual new data)
    new_data_scaled = scaler.transform(new_data)
    new_predictions = knn_regressor.predict(new_data_scaled)

    # Print the predictions on new data
    print('Predictions on new data:')
    print(new_predictions)
    return mse, new_predictions
