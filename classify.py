import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, BaggingClassifier, HistGradientBoostingRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# Example data

def getTrainingData(X,y,new_data):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# RANDOM FOREST REGRESSOR 
    
def test_train_split_RF(X,y, new_data):
    
    X_train, X_test, y_train, y_test = getTrainingData(X,y,new_data)


# Train a simple model (Replace with your actual model)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

# Make predictions on the test set
    y_pred = model.predict(X_test)

# Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

# Make predictions on a new array (Replace this array with your actual data)

    
    new_predictions = model.predict(new_data)
    print('Predictions on new data:')
    print(new_predictions)
    return mse, new_predictions
    
    
# BAGGING REGRESSOR
    
def test_train_split_BR(X,y, new_data):
    
    X_train, X_test, y_train, y_test = getTrainingData(X,y,new_data)


# Train a simple model (Replace with your actual model)
    model = BaggingRegressor()
    model.fit(X_train, y_train)

# Make predictions on the test set
    y_pred = model.predict(X_test)

# Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

# Make predictions on a new array (Replace this array with your actual data)


    new_predictions = model.predict(new_data)
    print('Predictions on new data:')
    print(new_predictions)
    return mse, new_predictions
    

# GRAIDENT BOOSTING

# def test_train_split_BC(X,y, new_data):
    
#     X_train, X_test, y_train, y_test = getTrainingData(X,y,new_data)


# # Train a simple model (Replace with your actual model)
#     model = BaggingClassifier()
#     model.fit(X_train, y_train)

# # Make predictions on the test set
#     y_pred = model.predict(X_test)

# # Evaluate the model
#     mse = mean_squared_error(y_test, y_pred)
#     print(f'Mean Squared Error: {mse}')

# # Make predictions on a new array (Replace this array with your actual data)


#     new_predictions = model.predict(new_data)
#     print('Predictions on new data:')
#     print(new_predictions)
    
    
    
# ADA BOOST REGRESSOR 
 
    
    
def test_train_split_AB(X,y, new_data):
    
    X_train, X_test, y_train, y_test = getTrainingData(X,y,new_data)


# Train a simple model (Replace with your actual model)
    model = AdaBoostRegressor()
    model.fit(X_train, y_train)

# Make predictions on the test set
    y_pred = model.predict(X_test)

# Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

# Make predictions on a new array (Replace this array with your actual data)


    new_predictions = model.predict(new_data)
    print('Predictions on new data:')
    print(new_predictions)
    


def test_train_split_HG(X,y, new_data):
    
    X_train, X_test, y_train, y_test = getTrainingData(X,y,new_data)


# Train a simple model (Replace with your actual model)
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)

# Make predictions on the test set
    y_pred = model.predict(X_test)

# Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

# Make predictions on a new array (Replace this array with your actual data)


    new_predictions = model.predict(new_data)
    print('Predictions on new data:')
    print(new_predictions)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def test_train_split_SVR(X, y, new_data):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = getTrainingData(X,y,new_data)
    print("yes")
    # Train a Support Vector Regressor (SVR) model
    model = SVR()
    
    print(X_train.shape)
    y_train_flat = y_train.flatten()
    y_train_flat = y_train_flat[:20]
    X_train_flat = X_train[:1,:]
    print(y_train_flat.shape)
    model.fit(X_train_flat, y_train_flat)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Make predictions on a new array
    new_predictions = model.predict(new_data)
    print('Predictions on new data:')
    print(new_predictions)

def run_tests(X,y,new_data):
    mse, new_predictions = test_train_split_RF(X,y, new_data)
    resultsRF = [mse, new_predictions]
    mse, new_predictions = test_train_split_BR(X,y, new_data)
    resultsBR = [mse, new_predictions]
    return resultsRF, resultsBR

    
    # mean squared error - 