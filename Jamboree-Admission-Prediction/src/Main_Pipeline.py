# =========================================
# JAMBOREE ADMISSION PREDICTION
# =========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")


# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv("../datasets/Jamboree_Admission.csv")


# =========================================
# BASIC EXPLORATION
# =========================================

print(df.info())

print(df.head())

print(df.describe())

print(df.isnull().sum())


# =========================================
# DROP SERIAL NUMBER
# =========================================

if "Serial No." in df.columns:
    df.drop("Serial No.", axis=1, inplace=True)


# =========================================
# TARGET DISTRIBUTION
# =========================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["Chance of Admit "],
    kde=True
)

plt.title("Admission Chance Distribution")

plt.savefig(
    "../screenshots/Admission_Chance_Distribution.png"
)

plt.show()


# =========================================
# CORRELATION HEATMAP
# =========================================

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "../screenshots/Correlation_Heatmap.png"
)

plt.show()


# =========================================
# GRE VS ADMISSION
# =========================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="GRE Score",
    y="Chance of Admit ",
    data=df
)

plt.title("GRE vs Admission Chance")

plt.savefig(
    "../screenshots/GRE_vs_Admission.png"
)

plt.show()


# =========================================
# TOEFL VS ADMISSION
# =========================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="TOEFL Score",
    y="Chance of Admit ",
    data=df
)

plt.title("TOEFL vs Admission Chance")

plt.savefig(
    "../screenshots/TOEFL_vs_Admission.png"
)

plt.show()


# =========================================
# CGPA VS ADMISSION
# =========================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="CGPA",
    y="Chance of Admit ",
    data=df
)

plt.title("CGPA vs Admission Chance")

plt.savefig(
    "../screenshots/CGPA_vs_Admission.png"
)

plt.show()


# =========================================
# MODEL BUILDING
# =========================================

X = df.drop(
    "Chance of Admit ",
    axis=1
)

y = df["Chance of Admit "]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# =========================================
# PREDICTIONS
# =========================================

y_pred = model.predict(X_test)


# =========================================
# MODEL EVALUATION
# =========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nMAE:", mae)

print("\nRMSE:", rmse)

print("\nR2 Score:", r2)


# =========================================
# ACTUAL VS PREDICTED
# =========================================

plt.figure(figsize=(8,5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title(
    "Actual vs Predicted Admission Chance"
)

plt.savefig(
    "../screenshots/Prediction_Analysis.png"
)

plt.show()


# =========================================
# FEATURE IMPORTANCE
# =========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)


plt.figure(figsize=(10,6))

sns.barplot(
    x="Coefficient",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")

plt.savefig(
    "../screenshots/Feature_Importance.png"
)

plt.show()


# =========================================
# EXPORT RESULTS
# =========================================

importance.to_csv(
    "../results/feature_importance.csv",
    index=False
)

print("\n============================")
print("JAMBOREE PIPELINE COMPLETED")
print("============================")