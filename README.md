# BMW Price Prediction and Visualization

This repository contains a Python script for predicting price trends and investment decisions for BMW M3 models. It also provides visualizations to analyze historical price data and trends across different models.

## Features
1. **Prediction**: Uses a Random Forest Classifier to predict whether to "Hold" or "Sell" an M3 model based on price changes.
2. **Visualization**: Generates various plots to understand pricing trends and relationships between key metrics.
3. **Custom Model Selection**: Allows users to focus on specific BMW M3 models and their production years.

## Input
The script reads data from a CSV file located at `bmw_price_predict/bmw_price-trends.csv`. Ensure the file contains columns like `Date`, `Price`, `Car Type`, `Avg Price`, `YoY Change`, etc.

## How It Works
1. **Data Preprocessing**:
   - Converts dates to datetime format.
   - Calculates price changes and filters data based on the selected M3 model's production years.

2. **Model Selection**:
   - Users select a specific BMW M3 model (e.g., "F80 M3") to analyze.
   - Filters the dataset for the chosen model.

3. **Prediction**:
   - Trains a Random Forest Classifier to decide whether to "Hold" or "Sell" the vehicle based on recent price trends.
   - Provides a prediction for the most recent data point.

4. **Visualization**:
   - Price distribution, trends over time, and correlations are visualized using Matplotlib, Seaborn, and Plotly.

## Output
- **Model Prediction**: Displays "Hold" or "Sell" decision based on the latest data.
- **Metrics**: Accuracy and classification report of the prediction model.
- **Visualizations**:
  - Histogram of prices.
  - Boxplot of average price by car type.
  - Correlation heatmap.
  - Line plots of price trends.
  - Scatterplots and bar plots to explore relationships in the data.

## Prerequisites
- Python 3.8+
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `seaborn`
  - `matplotlib`
  - `plotly`

Install the dependencies via:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib plotly
```
## Example
```
Available M3 models and their production years:
E30 M3: 1986-1991
E36 M3: 1992-1999
E46 M3: 2000-2006
E90/E92/E93 M3: 2007-2013
F80 M3: 2014-2019
G80 M3: 2020-2024
Enter the M3 model you want to predict (e.g., 'F80 M3'): F80 M3
Number of rows for F80 M3: 120
Accuracy: 0.87
              precision    recall  f1-score   support
           0       0.89      0.84      0.87        25
           1       0.86      0.90      0.88        25
Prediction for F80 M3: Hold
```

