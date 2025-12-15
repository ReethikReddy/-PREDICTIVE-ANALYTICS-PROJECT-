import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

#DATA LOADING & PREPARATION
df = pd.read_csv("C:\\Users\\hp\\Downloads\\TG Rainfall data November 2025.csv")

# Rename column
df.rename(columns={'Rain (mm)': 'Rainfall'}, inplace=True)

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nDATASET SHAPE:", df.shape)
print(df.head())

print("\nDATA INFO")
print(df.info())

print("\nSUMMARY STATISTICS")
print(df.describe())

# BASIC DATA VISUALIZATION
plt.figure(figsize=(6,4))
sns.histplot(df['Rainfall'], bins=30, kde=True)
plt.title("Rainfall Distribution")
plt.xlabel("Rainfall (mm)")
plt.show()

# ==================================================
# UNIT II – SUPERVISED LEARNING (REGRESSION)
# ==================================================
X_reg = df[['Min Humidity (%)', 'Max Humidity (%)']]
y_reg = df['Rainfall']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)

y_pred_r = reg_model.predict(X_test_r)

# ---------------- REGRESSION METRICS ----------------
print("\nREGRESSION RESULTS")
print("MAE :", mean_absolute_error(y_test_r, y_pred_r))
print("MSE :", mean_squared_error(y_test_r, y_pred_r))
print("RMSE:", np.sqrt(mean_squared_error(y_test_r, y_pred_r)))
print("R²  :", r2_score(y_test_r, y_pred_r))

# ---------------- REGRESSION PLOTS ----------------

# 1️⃣ Actual vs Predicted Plot
plt.figure(figsize=(6,4))
plt.scatter(y_test_r, y_pred_r, alpha=0.5)
plt.plot([0, max(y_test_r)], [0, max(y_test_r)], 'r--')
plt.xlabel("Actual Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("Actual vs Predicted Rainfall")
plt.show()

# 2️⃣ Residual Plot
residuals = y_test_r - y_pred_r

plt.figure(figsize=(6,4))
plt.scatter(y_pred_r, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Rainfall")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# ==================================================
# UNIT III – SUPERVISED LEARNING (CLASSIFICATION)
# ==================================================
df['High_Rainfall'] = (df['Rainfall'] > df['Rainfall'].mean()).astype(int)

X_cls = df[['Rainfall']]
y_cls = df['High_Rainfall']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

cls_model = LogisticRegression()
cls_model.fit(X_train_c, y_train_c)

y_pred_c = cls_model.predict(X_test_c)

print("\nCLASSIFICATION RESULTS")
print("Accuracy :", accuracy_score(y_test_c, y_pred_c))
print("Precision:", precision_score(y_test_c, y_pred_c))
print("Recall   :", recall_score(y_test_c, y_pred_c))
print("F1 Score :", f1_score(y_test_c, y_pred_c))

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_c)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==================================================
# UNIT IV – UNSUPERVISED LEARNING (CLUSTERING)
# ==================================================
X_cluster = df[['Rainfall', 'Min Humidity (%)', 'Max Humidity (%)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Cluster Visualization
plt.figure(figsize=(6,4))
sns.scatterplot(
    x=df['Rainfall'],
    y=df['Max Humidity (%)'],
    hue=df['Cluster'],
    palette='Set1'
)
plt.title("K-Means Clustering (Rainfall vs Max Humidity)")
plt.show()

print("\n===== PROGRAM COMPLETED SUCCESSFULLY =====")

