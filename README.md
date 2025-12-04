# Coursework-ML
Import Libraries
# Core libraries
import pandas as pd
import numpy as np

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Metrics
from sklearn.metrics import mean_squared_error, r2_score

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
# Paths in Google Colab
attribute_path = "/content/Attribute DataSet.xlsx"
sales_path = "/content/Dress Sales.xlsx"

# Load data
attr_df = pd.read_excel(attribute_path)
sales_df = pd.read_excel(sales_path)

print("Attribute Data:")
display(attr_df.head())

print("Sales Data:")
display(sales_df.head())
# Remove Dress_ID column and convert remaining values to numeric
sales_values = sales_df.drop(columns=["Dress_ID"]).apply(pd.to_numeric, errors="coerce")

# Add total sales column
sales_df["Total_Sales"] = sales_values.sum(axis=1)

sales_df_clean = sales_df[["Dress_ID", "Total_Sales"]]

display(sales_df_clean.head())
merged_df = pd.merge(attr_df, sales_df_clean, on="Dress_ID", how="inner")
print("Merged dataframe shape:", merged_df.shape)
display(merged_df.head())
# Identify categorical columns
categorical_cols = merged_df.select_dtypes(include=["object"]).columns

# Apply Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    merged_df[col] = le.fit_transform(merged_df[col].astype(str))

display(merged_df.head())
# Independent variables
X = merged_df.drop(columns=["Dress_ID", "Total_Sales"])

# Target variable
y = merged_df["Total_Sales"]

# Standardisation (for SVR)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

y_pred_svr = svr.predict(X_test)

rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
r2_svr = r2_score(y_test, y_pred_svr)

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "SVR"],
    "RMSE": [rmse_lr, rmse_rf, rmse_svr],
    "R2 Score": [r2_lr, r2_rf, r2_svr]
})

display(results)

importances = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Random Forest: Actual vs Predicted Sales")
plt.grid(True)
plt.show()
