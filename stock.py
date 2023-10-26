import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_and_process_data(filepath):
    """
    Load and process the stock data from a CSV file.
    """
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Volume'] = data['Volume'].str.replace(',', '').astype(int)
    data.sort_values('Date', inplace=True)

    return data

def exploratory_data_analysis(data):
    """
    Conduct exploratory data analysis and print summary statistics.
    """
    print("Summary Statistics:")
    print(data.describe())
    print("\nGeneral Information:")
    print(data.info())

def calculate_statistical_measures(data):
    """
    Calculate statistical measures such as daily returns and moving averages.
    """
    data['Daily Return'] = data['Close'].pct_change()
    data['Moving Average (50 days)'] = data['Close'].rolling(window=50).mean()

def visualize_data(data):
    """
    Create visualizations for the stock data trends and statistics.
    """
    # Plot of closing prices and moving average
    plt.figure(figsize=(14,7))
    plt.plot(data['Date'], data['Close'], label='Closing Prices', alpha=0.5)
    plt.plot(data['Date'], data['Moving Average (50 days)'], label='50-day Moving Average')
    plt.title('Closing Stock Prices and 50-day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price (in USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Histogram of daily returns
    sns.histplot(data['Daily Return'].dropna(), bins=100, color='purple')
    plt.title('Histogram of Daily Returns')
    plt.show()

def implement_machine_learning(data):
    """
    Stretch Goal: Implement a basic machine learning model to predict future stock prices.
    """
    # Preprocessing: Create a column of previous closing prices shifted by 1
    data['Previous Close'] = data['Close'].shift(1)

    # Drop the NaN values created by shift operation
    data = data.dropna()

    # Feature selection: Select 'Previous Close' as feature and 'Close' as the target variable
    X = data[['Previous Close']]
    y = data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate and print metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Plot actual vs. predicted values
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Previous Close Price')
    plt.ylabel('Predicted Stock Price')
    plt.show()

def main():
    """
    Main function to execute the data analysis process.
    """
    # Ask the user for the file path
    # This allows the user to decide which data they would like to analyze
    filename = input("Please enter the name of your stock data file (without extension): ")

    # Append '.csv' extension to the file name
    filepath = f"{filename}.csv"

    # Data loading and processing
    data = load_and_process_data(filepath)

    # Question 1: What are the trends in stock prices over the specified period?
    # Exploratory data analysis
    exploratory_data_analysis(data)

    # Calculate statistical measures
    calculate_statistical_measures(data)

    # Visualize the data trends and patterns
    visualize_data(data)

    # Question 2: What are the significant changes or anomalies in stock performance?
    # This question can be addressed through the exploratory analysis and the daily returns calculation.

    # Question 3: How do the stock prices correlate with the previous day's prices?
    # [Stretch Goal] Implement machine learning prediction or classification here, if applicable
    implement_machine_learning(data)

if __name__ == "__main__":
    main()