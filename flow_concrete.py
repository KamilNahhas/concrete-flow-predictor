import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load .csv dataset
df = pd.read_csv('slump_test.data')

# Separate dataframe into attributes and target variable
df_x = df[["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr.", "Fine Aggr."]]
df_y = df[["FLOW(cm)"]]

# Convert to numpy array
flow_X = df_x.to_numpy()
flow_y = df_y.to_numpy()

# Split the data into training/testing sets
flow_X_train = flow_X[:-20]
flow_X_test = flow_X[-20:]
# Split the target into training/testing sets
flow_y_train = flow_y[:-20]
flow_y_test = flow_y[-20:]

print("Welcome to the concrete flow predictor")
print("The prediction is made by using multiple linear regression")
method = int(input("Select which method you want (0 = by hand, 1 = framework): "))
print("Building model...\n")

if method == 1:

    # # # # # # # # # # # # # # # # # #
    # LINEAR REGRESSION BY FRAMEWORK  #
    # # # # # # # # # # # # # # # # # #

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(flow_X_train, flow_y_train)

    # Make predictions using the testing set
    flow_y_pred = regr.predict(flow_X_test)
    beta = regr.coef_

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(flow_y_test, flow_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(flow_y_test, flow_y_pred))
else:

    # # # # # # # # # # # # # # #
    # LINEAR REGRESSION BY HAND #
    # # # # # # # # # # # # # # #

    beta = np.dot(np.dot(np.linalg.inv(np.dot(flow_X_train.T, flow_X_train)), flow_X_train.T), flow_y_train)
    flow_y_pred = np.dot(flow_X_test, beta)

    SSE = np.square(np.subtract(flow_y_test, flow_y_pred)).sum()
    SST = np.square(np.subtract(flow_y_test, flow_y_test.mean())).sum()
    MSE = np.square(np.subtract(flow_y_test, flow_y_pred)).mean()
    r2 = 1 - SSE / SST

    # The coefficients
    beta = beta.T
    print('Coefficients: \n', beta)
    # The mean squared error
    print('Mean squared error: %.2f'
          % MSE)
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2)

print("\n")
print("Now it is time to predict the flow of concrete!\n")
print("Input the data separated by commas and in order of")
print("Cement, Slag, Fly ash, Water, SP, Coarse Aggr., Fine Aggr.")
print("Remember that this data is expressed in component kg in one m^3 of concrete\n")

cycle = 1
while cycle == 1:
    flow_X_input = np.array([list(map(float, input("Input: ").split(',')))])
    flow_y_input_pred = np.dot(flow_X_input, beta.T)
    print('Prediction: %.2f'
          % flow_y_input_pred)
    cycle = int(input("Would you like to make another prediction (0 = No, 1 = Yes): "))


