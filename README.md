# Ex.No: 6               HOLT WINTERS METHOD
### Date: 
### NAME: MOHAMED HAMEEM SAJITH J
### REG NO: 212223240090


### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

try:
    df_full = pd.read_csv('GlobalLandTemperaturesByCity.csv')
except FileNotFoundError:
    print("Error: 'GlobalLandTemperaturesByCity.csv' not found.")
    print("Please download the file from the Kaggle link and place it in the same directory.")
    exit()

df = df_full[df_full['City'] == 'Ahmadabad'].copy()

if df.empty:
    print("Error: No data found for the specified city. Please check the city name.")
else:
    df['Date'] = pd.to_datetime(df['dt'])

    df = df.set_index('Date')

    df = df[['AverageTemperature']]

    df = df.dropna()

    print("First five rows of the filtered 'Ahmadabad' data:")
    print(df.head())
    print("\n")

    data_monthly = df['AverageTemperature'].resample('MS').mean()

    data_monthly = data_monthly.interpolate()

    print("Displaying plot of Monthly Average Temperature...")
    plt.figure(figsize=(12, 6))
    data_monthly.plot()
    plt.title('Monthly Average Temperature for Ahmadabad')
    plt.xlabel('Year')
    plt.ylabel('Temperature (C)')
    plt.grid(True)
    plt.show()

    print("Displaying Decomposition Plot...")
    decomposition = seasonal_decompose(data_monthly, model='additive', period=12)
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    plt.suptitle('Decomposition Plot', y=1.02)
    plt.tight_layout()
    plt.show()


    train_size = int(len(data_monthly) * 0.8)
    train_data = data_monthly[:train_size]
    test_data = data_monthly[train_size:]

    print(f"Training data size: {len(train_data)}")
    print(f"Testing data size: {len(test_data)}")
    print("\n")

    model_add = ExponentialSmoothing(
        train_data,
        trend='add',
        seasonal='add',
        seasonal_periods=12
    )

    fitted_model_add = model_add.fit()

    test_predictions_add = fitted_model_add.forecast(steps=len(test_data))

    print("Displaying Test vs. Prediction Plot (Additive Model)...")
    plt.figure(figsize=(12, 6))
    train_data.plot(label='Train Data')
    test_data.plot(label='Test Data', color='orange')
    test_predictions_add.plot(label='Test Predictions', color='green', linestyle='--')
    plt.title('Holt-Winters (Additive) - Test vs. Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

    rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print("\n")


    final_model = ExponentialSmoothing(
        data_monthly,  # Train on all data
        trend='add',
        seasonal='mul',
        seasonal_periods=12
    )

    final_fitted_model = final_model.fit()

    future_predictions = final_fitted_model.forecast(steps=36)

    print("Displaying Final Forecast Plot...")
    plt.figure(figsize=(12, 6))
    data_monthly.plot(label='Historical Data')
    future_predictions.plot(label='Future Forecast', color='red', linestyle='--')
    plt.title('Final Holt-Winters Forecast (3 Years)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Experiment 6 Complete.")
```

### OUTPUT:

<img width="1010" height="547" alt="image" src="https://github.com/user-attachments/assets/01bb94ce-f133-432a-819e-4aae02c9c9b8" />


### TEST_PREDICTION:
<img width="990" height="547" alt="image" src="https://github.com/user-attachments/assets/e13c3320-6e60-4534-8d03-6932136424d8" />
Root Mean Squared Error (RMSE): 1.0616


### DECOMPOSITION:
<img width="989" height="820" alt="image" src="https://github.com/user-attachments/assets/117a0088-98ed-4e5a-9f8e-6c0088a66ab2" />
Training data size: 2089
Testing data size: 523

### FINAL_PREDICTION :

<img width="990" height="547" alt="image" src="https://github.com/user-attachments/assets/cc1e54b8-31d5-4a80-9505-ee36706f591c" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
