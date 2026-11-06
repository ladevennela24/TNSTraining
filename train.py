# Step 1: Import libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load dataset
df = pd.read_csv("synthetic_online_retail_data.csv")

# Step 3: Explore dataset
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Step 4: Preprocess data
df["order_date"] = pd.to_datetime(
    df["order_date"], errors="coerce"
)
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

df = df.dropna(subset=["order_date", "quantity"])

df["Year"] = df["order_date"].dt.year
df["Month"] = df["order_date"].dt.month
df["Day"] = df["order_date"].dt.day

# Step 5: Split features and target
X = df.drop(columns=["order_date", "quantity"])
y = df["quantity"]

X = pd.get_dummies(X, drop_first=True)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).fillna(0)

valid = np.isfinite(y)
X = X.loc[valid]
y = y.loc[valid]

# Step 6: Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred1 = lr.predict(X_test)

print("\nLinear Regression")
print("MSE:", mean_squared_error(y_test, pred1))
print("R2 Score:", r2_score(y_test, pred1))

# Step 8: Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred2 = rf.predict(X_test)

print("\nRandom Forest")
print("MSE:", mean_squared_error(y_test, pred2))
print("R2 Score:", r2_score(y_test, pred2))

# Step 9: Select and save the best model
if mean_squared_error(y_test, pred1) < mean_squared_error(y_test, pred2):
    best_model = lr
    print("\nBest Model: Linear Regression")
else:
    best_model = rf
    print("\nBest Model: Random Forest")

joblib.dump(best_model, "best_retail_model.pkl")
print("Model saved successfully!")

# Step 10: Save test predictions
predictions = best_model.predict(X_test)

result = pd.DataFrame({
    "Actual Quantity": y_test,
    "Predicted Quantity": predictions
})

result.to_csv("sales_predictions.csv", index=False)
print("Predictions saved successfully!")
print(result.head())