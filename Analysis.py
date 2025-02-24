import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Path to the Excel file
file_path = '/Users/lathanaganur/Downloads/wework_data_cleaned.xlsx'

# Read the Excel file
df_historical = pd.read_excel(file_path, sheet_name='Historical_Data')
df_revenue_occupancy = pd.read_excel(file_path, sheet_name='Revenue_Occupancy')
df_employees = pd.read_excel(file_path, sheet_name='Employees_Data')
df_stock = pd.read_excel(file_path, sheet_name='Stock_Data')
df_revenue_projections = pd.read_excel(file_path, sheet_name='Revenue_Projections')
df_occupancy_projections = pd.read_excel(file_path, sheet_name='Occupancy_Projections')

# Plot Revenue Growth Over Time
plt.figure(figsize=(10, 6))
plt.plot(df_historical["Year"], df_historical["Revenue"], marker='o')
plt.title("WeWork Revenue Growth (2010-2023)")
plt.xlabel("Year")
plt.ylabel("Revenue (in million USD)")
plt.grid(True)
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_revenue_growth_over_years.png")
plt.show()

# Plot Occupancy Rates Over Time
plt.figure(figsize=(10, 6))
plt.plot(df_historical["Year"], df_historical["Occupancy_Rate"], marker='o', color='orange')
plt.title("WeWork Occupancy Rates (2010-2023)")
plt.xlabel("Year")
plt.ylabel("Occupancy Rate (%)")
plt.grid(True)
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_occupancy_rates_over_years.png")
plt.show()

# Plot Net Income Over Time
plt.figure(figsize=(10, 6))
plt.plot(df_historical["Year"], df_historical["Net_Income"], marker='o', color='red')
plt.title("WeWork Net Income (2010-2023)")
plt.xlabel("Year")
plt.ylabel("Net Income (in million USD)")
plt.grid(True)
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_net_income_over_years.png")
plt.show()

# Descriptive Analysis Graphs
# Revenue Growth (2016-2022)
years = df_revenue_occupancy["Year"].values
revenue = df_revenue_occupancy["Revenue"].values

plt.figure(figsize=(10, 6))
plt.plot(years, revenue, marker='o', linestyle='-', color='b', label='Revenue')
plt.title('WeWork Revenue Growth (2016-2022)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Revenue (in million USD)', fontsize=12)
plt.grid(True)
plt.legend()
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_revenue_growth_over_years_detailed.png", facecolor='w')
plt.show()

# Occupancy Rates (2016-2022)
occupancy_rates = df_revenue_occupancy["Occupancy_Rate"].values

plt.figure(figsize=(10, 6))
plt.plot(years, occupancy_rates, marker='o', linestyle='-', color='orange', label='Occupancy Rate')
plt.title('WeWork Occupancy Rates (2016-2022)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Occupancy Rate (%)', fontsize=12)
plt.grid(True)
plt.legend()
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_occupancy_rates_over_years_detailed.png", facecolor='w')
plt.show()

# Number of Employees (2020 vs 2022)
years_employees = df_employees["Year"].values
num_employees = df_employees["Number_of_Employees"].values

plt.figure(figsize=(10, 6))
plt.bar(years_employees, num_employees, color=['blue', 'green'])
plt.title('Number of Employees at WeWork (2020 vs 2022)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.ylim(0, 5000)
plt.grid(axis='y')
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_employees_over_years_detailed.png", facecolor='w')
plt.show()

# Diagnostic Analysis Graphs
# Revenue and Occupancy Correlation
plt.figure(figsize=(10, 6))
plt.scatter(revenue, occupancy_rates, color='red')
plt.title('Revenue and Occupancy Rate Correlation', fontsize=14)
plt.xlabel('Revenue (in million USD)', fontsize=12)
plt.ylabel('Occupancy Rate (%)', fontsize=12)
plt.grid(True)
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/revenue_occupancy_correlation.png", facecolor='w')
plt.show()

# Predictive Analysis Graphs
# Revenue Projections (2023-2025)
future_years = df_revenue_projections["Year"].values
revenue_projections = df_revenue_projections["Revenue_Projection"].values

plt.figure(figsize=(10, 6))
plt.plot(np.append(years, future_years), np.append(revenue, revenue_projections), marker='o', linestyle='-', color='purple', label='Revenue Projection')
plt.title('WeWork Revenue Projections (2023-2025)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Revenue (in million USD)', fontsize=12)
plt.grid(True)
plt.legend()
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/revenue_projections_2023_2025.png", facecolor='w')
plt.show()

# Occupancy Rate Projections (2023-2025)
occupancy_projections = df_occupancy_projections["Occupancy_Rate_Projection"].values

plt.figure(figsize=(10, 6))
plt.plot(np.append(years, future_years), np.append(occupancy_rates, occupancy_projections), marker='o', linestyle='-', color='teal', label='Occupancy Rate Projection')
plt.title('WeWork Occupancy Rate Projections (2023-2025)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Occupancy Rate (%)', fontsize=12)
plt.grid(True)
plt.legend()
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/occupancy_projections_2023_2025.png", facecolor='w')
plt.show()

# Stock Data Analysis
# Convert 'Date' column to datetime
df_stock['Date'] = pd.to_datetime(df_stock['Date'])

# Set 'Date' as the index
df_stock.set_index('Date', inplace=True)

# Prepare data for predictive analysis
X = np.arange(len(df_stock)).reshape(-1, 1)  # Index as feature
y = df_stock['Close'].values  # Close prices as target

# Split the data into training and testing sets
X_train, X_test = X[:-1], X[-1:]
y_train, y_test = y[:-1], y[-1:]

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the future prices
y_pred = model.predict(X_test.reshape(-1, 1))

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(df_stock.index, df_stock['Close'], marker='o', linestyle='-', color='b', label='Actual')
plt.plot([df_stock.index[-1]], y_pred, marker='x', linestyle='--', color='r', label='Predicted')
plt.title('WeWork Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_stock_prediction_5_years.png", facecolor='w')
plt.show()

# Print the Mean Squared Error
print(f"Mean Squared Error: {mse}")
print(f"Predicted Value: {y_pred}")

# Prescriptive Analysis Graphs
# Strategic Recommendations Overview
strategic_recommendations = ['Expand Flexible Options', 'Leverage Technology', 'New Market Entry']
impact = [8, 7, 9]  # Example impact score out of 10

plt.figure(figsize=(10, 6))
plt.bar(strategic_recommendations, impact, color='magenta')
plt.title('Strategic Recommendations Impact', fontsize=14)
plt.xlabel('Strategy', fontsize=12)
plt.ylabel('Impact Score', fontsize=12)
plt.ylim(0, 10)
plt.grid(axis='y')
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/strategic_recommendations.png", facecolor='w')
plt.show()
