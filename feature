# Feature Scaling (Standardization and Normalization)

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---- Create Dataset ----
data = {
"Country":["France","Spain","Germany","Spain","Germany","France","Spain","France","Germany","France"],
"Age":[44,27,30,38,40,35,31,48,50,37],
"Salary":[72000,48000,54000,61000,85000,58000,52000,79000,83000,67000],
"Purchased":["No","Yes","No","No","Yes","Yes","No","Yes","No","Yes"]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# ---- Select Numerical Features ----
num_features = df[["Age","Salary"]]

# ---- Standardization ----
scaler_standard = StandardScaler()
standardized_data = scaler_standard.fit_transform(num_features)

standardized_df = pd.DataFrame(standardized_data, columns=num_features.columns)

print("\nStandardized Data:\n")
print(standardized_df)

# ---- Normalization (Min-Max Scaling) ----
scaler_minmax = MinMaxScaler()
normalized_data = scaler_minmax.fit_transform(num_features)

normalized_df = pd.DataFrame(normalized_data, columns=num_features.columns)

print("\nNormalized Data:\n")
print(normalized_df)
