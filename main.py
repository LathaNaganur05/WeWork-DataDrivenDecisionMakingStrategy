
import matplotlib.pyplot as plt
import pandas as pd

# Sample historical data for WeWork based on general insights and trends mentioned
data = {
    "Year": [2010, 2012, 2014, 2016, 2018, 2019, 2020, 2021, 2022, 2023],
    "Revenue": [5, 100, 300, 700, 1820, 3460, 3420, 2570, 3250, 3360],  # in millions USD
    "Occupancy_Rate": [50, 60, 65, 70, 75, 80, 68, 70, 73, 72],  # in percentage
    "Net_Income": [-5, -50, -150, -300, -1500, -2200, -3200, -2000, -1800, -1500]  # in millions USD
}

# Create DataFrame
df = pd.DataFrame(data)

# Plot Revenue Growth Over Time
plt.figure(figsize=(10, 6))
plt.plot(df["Year"], df["Revenue"], marker='o')
plt.title("WeWork Revenue Growth (2010-2023)")
plt.xlabel("Year")
plt.ylabel("Revenue (in million USD)")
plt.grid(True)
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_revenue_growth_over_years.png")
plt.show()

# Plot Occupancy Rates Over Time
plt.figure(figsize=(10, 6))
plt.plot(df["Year"], df["Occupancy_Rate"], marker='o', color='orange')
plt.title("WeWork Occupancy Rates (2010-2023)")
plt.xlabel("Year")
plt.ylabel("Occupancy Rate (%)")
plt.grid(True)
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_occupancy_rates_over_years.png")
plt.show()

# Plot Net Income Over Time
plt.figure(figsize=(10, 6))
plt.plot(df["Year"], df["Net_Income"], marker='o', color='red')
plt.title("WeWork Net Income (2010-2023)")
plt.xlabel("Year")
plt.ylabel("Net Income (in million USD)")
plt.grid(True)
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_net_income_over_years.png")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Descriptive Analysis Graphs

# Revenue Growth (2016-2022)
years = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022])
revenue = np.array([436, 886, 1800, 3440, 2740, 2900, 3200])  # Example data in millions USD

plt.figure(figsize=(10, 6))
plt.plot(years, revenue, marker='o', linestyle='-', color='b', label='Revenue')
plt.title('WeWork Revenue Growth (2016-2022)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Revenue (in million USD)', fontsize=12)
plt.grid(True)
plt.legend()
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_revenue_growth_over_years_detailed.png", facecolor='w')

# Occupancy Rates (2016-2022)
occupancy_rates = np.array([68, 72, 75, 80, 65, 70, 73])  # Example data in percentage

plt.figure(figsize=(10, 6))
plt.plot(years, occupancy_rates, marker='o', linestyle='-', color='orange', label='Occupancy Rate')
plt.title('WeWork Occupancy Rates (2016-2022)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Occupancy Rate (%)', fontsize=12)
plt.grid(True)
plt.legend()
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_occupancy_rates_over_years_detailed.png", facecolor='w')

# Number of Employees (2020 vs 2022)
years_employees = ['2020', '2022']
num_employees = [4700, 4300]

plt.figure(figsize=(10, 6))
plt.bar(years_employees, num_employees, color=['blue', 'green'])
plt.title('Number of Employees at WeWork (2020 vs 2022)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.ylim(0, 5000)
plt.grid(axis='y')
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/wework_employees_over_years_detailed.png", facecolor='w')

# Diagnostic Analysis Graphs

# Revenue and Occupancy Correlation
plt.figure(figsize=(10, 6))
plt.scatter(revenue, occupancy_rates, color='red')
plt.title('Revenue and Occupancy Rate Correlation', fontsize=14)
plt.xlabel('Revenue (in million USD)', fontsize=12)
plt.ylabel('Occupancy Rate (%)', fontsize=12)
plt.grid(True)
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/revenue_occupancy_correlation.png", facecolor='w')

# Predictive Analysis Graphs

# Revenue Projections (2023-2025)
future_years = np.array([2023, 2024, 2025])
revenue_projections = np.array([3500, 3800, 4100])  # Example projected data in millions USD

plt.figure(figsize=(10, 6))
plt.plot(np.append(years, future_years), np.append(revenue, revenue_projections), marker='o', linestyle='-', color='purple', label='Revenue Projection')
plt.title('WeWork Revenue Projections (2023-2025)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Revenue (in million USD)', fontsize=12)
plt.grid(True)
plt.legend()
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/revenue_projections_2023_2025.png", facecolor='w')

# Occupancy Rate Projections (2023-2025)
occupancy_projections = np.array([75, 77, 79])  # Example projected data in percentage

plt.figure(figsize=(10, 6))
plt.plot(np.append(years, future_years), np.append(occupancy_rates, occupancy_projections), marker='o', linestyle='-', color='teal', label='Occupancy Rate Projection')
plt.title('WeWork Occupancy Rate Projections (2023-2025)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Occupancy Rate (%)', fontsize=12)
plt.grid(True)
plt.legend()
plt.savefig("/Users/lathanaganur/PycharmProjects/DDDM/occupancy_projections_2023_2025.png", facecolor='w')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data for WeWork stock (based on the provided screenshot)
data = {
    "Date": ["2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
    "Close": [600, 500, 300, 100, 0.06, 0.03],
    "Volume": [6000000, 5000000, 4000000, 3000000, 2000000, 1000000]
}
df_stock = pd.DataFrame(data)

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
mse, y_pred, "/Users/lathanaganur/PycharmProjects/DDDM/wework_stock_prediction_5_years.png"


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
