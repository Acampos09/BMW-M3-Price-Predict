import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV file
file_name = 'bmw_price_predict/bmw_price-trends.csv'
df = pd.read_csv(file_name)

# Preprocess the data
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.dropna()

# Calculate price changes
df['Price Change'] = df['Price'].pct_change()
df['YoY Change'] = df['YoY Change'].apply(pd.to_numeric, errors='coerce')
df['Year'] = df['Car Type'].str.extract(r'(\d{4})').astype(int)
df['Model'] = df['Car Type'].str.extract(r'(M3)')

# Define M3 models and their production years
m3_models = {
    "E30 M3": (1986, 1991),
    "E36 M3": (1992, 1999),
    "E46 M3": (2000, 2006),
    "E90/E92/E93 M3": (2007, 2013),
    "F80 M3": (2014, 2019),
    "G80 M3": (2020, 2024)
}

# Show available models to the user
print("Available M3 models and their production years:")
for model, years in m3_models.items():
    print(f"{model}: {years[0]}-{years[1]}")

# User selects a model
selected_model = input("Enter the M3 model you want to predict (e.g., 'F80 M3'): ").strip()
if selected_model not in m3_models:
    print("Invalid model selected. Please restart the script and choose a valid model.")
    exit()

# Filter data based on the selected model
start_year, end_year = m3_models[selected_model]
model_data = df[(df['Year'] >= start_year) & (df['Year'] <= end_year) & (df['Model'] == 'M3')]

if model_data.empty:
    print(f"No data available for {selected_model}.")
    exit()

print(f"Number of rows for {selected_model}: {len(model_data)}")

# Create target column: Decision (1 = hold, 0 = sell)
model_data.loc[:,'Decision'] = model_data['Price Change'].apply(lambda x: 1 if x > 0 else 0)

# One-hot encode 'Car Type' for the selected model
model_data_encoded = pd.get_dummies(model_data, columns=['Car Type'], drop_first=True)

# Prepare features and target
features = ['Avg Price', 'Last 30 days', 'Last 90 days', 'YoY Change']
X = model_data_encoded[features]
y = model_data_encoded['Decision']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred, zero_division=1))

# Predict the decision for the most recent data
latest_data = model_data_encoded.tail(1)[features]
prediction = model.predict(latest_data)
decision = 'Hold' if prediction[0] == 1 else 'Sell'
print(f'Prediction for {selected_model}: {decision}')

# Optional: Set Seaborn style for cleaner visuals
sns.set(style="whitegrid")

# Histogram of Price
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=30, kde=True, color='blue')
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Boxplot for Avg Price by Car Type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Car Type', y='Avg Price')
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.title('Avg Price by Car Type')
plt.show()


# Price over time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], color='green')
plt.title('Price Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# Pair plot to explore correlations between numerical columns
num_cols = ['Price', 'Avg Price', 'Last 30 days', 'Last 90 days', 'YoY Change']
sns.pairplot(df[num_cols])
plt.show()


# Correlation heatmap
plt.figure(figsize=(10, 6))
correlation = df[['Price', 'Avg Price', 'Last 30 days', 'Last 90 days', 'YoY Change']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# Bar plot of Avg Price for each Car Type
plt.figure(figsize=(12, 6))
avg_price_by_type = df.groupby('Car Type')['Avg Price'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_by_type.index, y=avg_price_by_type.values, palette='viridis')
plt.xticks(rotation=45)
plt.title('Average Price by Car Type')
plt.ylabel('Average Price')
plt.show()


# Scatter plot: Price vs. YoY Change
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='YoY Change', y='Price', hue='Car Type', palette='muted')
plt.title('Price vs. YoY Change')
plt.xlabel('YoY Change')
plt.ylabel('Price')
plt.show()

# Visualize the price trend for the selected model
fig = px.line(model_data, x='Date', y='Price', title=f'{selected_model} Price Trends Over Time')
fig.show()

fig = px.line(
    df, 
    x='Date', 
    y='Price', 
    color='Car Type', 
    title='Price Trend for All Vehicles',
    labels={'Date': 'Date', 'Price': 'Price (USD)', 'Car Type': 'Vehicle Type'}
)

# Display the chart
fig.show()


