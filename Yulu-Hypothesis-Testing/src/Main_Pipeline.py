# =========================================
# YULU HYPOTHESIS TESTING & DEMAND ANALYSIS
# =========================================

# =========================================
# IMPORT REQUIRED LIBRARIES
# =========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    ttest_ind,
    f_oneway,
    chi2_contingency,
    levene,
    shapiro
)

import warnings

warnings.filterwarnings("ignore")


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("../datasets/yulu_data.csv")


# =========================================
# BASIC DATA EXPLORATION
# =========================================

print("\n================ DATASET INFO ================\n")

print(df.info())


print("\n================ FIRST 5 ROWS ================\n")

print(df.head())


print("\n================ LAST 5 ROWS ================\n")

print(df.tail())


print("\n================ DATASET SHAPE ================\n")

print(df.shape)


print("\n================ COLUMN NAMES ================\n")

print(df.columns)


print("\n================ STATISTICAL SUMMARY ================\n")

print(df.describe())


# =========================================
# MISSING VALUE ANALYSIS
# =========================================

print("\n================ MISSING VALUES ================\n")

print(df.isnull().sum())


# =========================================
# DUPLICATE VALUE ANALYSIS
# =========================================

print("\n================ DUPLICATE RECORDS ================\n")

print(df.duplicated().sum())


# =========================================
# DATA TYPE CONVERSION
# =========================================

categorical_columns = [
    'season',
    'holiday',
    'workingday',
    'weather'
]


for col in categorical_columns:
    
    df[col] = df[col].astype('category')


# =========================================
# DATETIME FEATURE ENGINEERING
# =========================================

df['datetime'] = pd.to_datetime(df['datetime'])


df['year'] = df['datetime'].dt.year

df['month'] = df['datetime'].dt.month

df['day'] = df['datetime'].dt.day

df['hour'] = df['datetime'].dt.hour

df['weekday'] = df['datetime'].dt.day_name()


# =========================================
# UNIVARIATE ANALYSIS
# =========================================

# Count Distribution

plt.figure(figsize=(10, 5))

sns.histplot(
    df['count'],
    bins=50,
    kde=True
)

plt.title("Bike Rental Count Distribution")

plt.savefig(
    "../screenshots/Bike_Rental_Count_Distribution.png"
)

plt.show()


# Temperature Distribution

plt.figure(figsize=(10, 5))

sns.histplot(
    df['temp'],
    bins=40,
    kde=True
)

plt.title("Temperature Distribution")

plt.savefig(
    "../screenshots/Temperature_Distribution.png"
)

plt.show()


# Humidity Distribution

plt.figure(figsize=(10, 5))

sns.histplot(
    df['humidity'],
    bins=40,
    kde=True
)

plt.title("Humidity Distribution")

plt.savefig(
    "../screenshots/Humidity_Distribution.png"
)

plt.show()


# Windspeed Distribution

plt.figure(figsize=(10, 5))

sns.histplot(
    df['windspeed'],
    bins=40,
    kde=True
)

plt.title("Windspeed Distribution")

plt.savefig(
    "../screenshots/Windspeed_Distribution.png"
)

plt.show()


# =========================================
# OUTLIER ANALYSIS
# =========================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df['count']
)

plt.title("Bike Rental Count Outlier Analysis")

plt.savefig(
    "../screenshots/Bike_Rental_Outlier_Analysis.png"
)

plt.show()


# =========================================
# BIVARIATE ANALYSIS
# =========================================

# Working Day vs Count

plt.figure(figsize=(10, 5))

sns.boxplot(
    x='workingday',
    y='count',
    data=df
)

plt.title("Working Day vs Bike Rentals")

plt.savefig(
    "../screenshots/Workingday_vs_Count.png"
)

plt.show()


# Season vs Count

plt.figure(figsize=(10, 5))

sns.boxplot(
    x='season',
    y='count',
    data=df
)

plt.title("Season vs Bike Rentals")

plt.savefig(
    "../screenshots/Season_vs_Count.png"
)

plt.show()


# Weather vs Count

plt.figure(figsize=(10, 5))

sns.boxplot(
    x='weather',
    y='count',
    data=df
)

plt.title("Weather vs Bike Rentals")

plt.savefig(
    "../screenshots/Weather_vs_Count.png"
)

plt.show()


# Hourly Demand Analysis

hourly_count = (
    df.groupby('hour')['count']
    .mean()
)


plt.figure(figsize=(12, 6))

hourly_count.plot()

plt.title("Hourly Bike Rental Demand")

plt.xlabel("Hour")

plt.ylabel("Average Rentals")

plt.savefig(
    "../screenshots/Hourly_Bike_Rental_Demand.png"
)

plt.show()


# =========================================
# CORRELATION ANALYSIS
# =========================================

numerical_columns = df.select_dtypes(
    include=np.number
).columns


correlation_matrix = (
    df[numerical_columns]
    .corr()
)


plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    cmap='coolwarm',
    annot=True
)

plt.title("Correlation Heatmap")

plt.savefig(
    "../screenshots/Correlation_Heatmap.png"
)

plt.show()


# =========================================
# HYPOTHESIS TESTING
# =========================================

# =========================================
# T-TEST
# =========================================

print("\n================ T-TEST ANALYSIS ================\n")

working_day = df[
    df['workingday'] == 1
]['count']


non_working_day = df[
    df['workingday'] == 0
]['count']


# Levene Test

levene_stat, levene_p = levene(
    working_day,
    non_working_day
)


print("\nLevene Test P-Value:")

print(levene_p)


# T-Test

t_stat, p_value = ttest_ind(
    working_day,
    non_working_day,
    equal_var=False
)


print("\nT-Test Statistic:")

print(t_stat)


print("\nT-Test P-Value:")

print(p_value)


if p_value < 0.05:
    
    print("\nReject Null Hypothesis")
    
    print(
        "Working Day has significant effect on bike rentals"
    )

else:
    
    print("\nFail to Reject Null Hypothesis")
    
    print(
        "Working Day does not significantly affect bike rentals"
    )


# =========================================
# ANOVA TEST - SEASON
# =========================================

print("\n================ ANOVA TEST - SEASON ================\n")


season_1 = df[
    df['season'] == 1
]['count']


season_2 = df[
    df['season'] == 2
]['count']


season_3 = df[
    df['season'] == 3
]['count']


season_4 = df[
    df['season'] == 4
]['count']


anova_stat, anova_p = f_oneway(
    season_1,
    season_2,
    season_3,
    season_4
)


print("\nANOVA Statistic:")

print(anova_stat)


print("\nANOVA P-Value:")

print(anova_p)


if anova_p < 0.05:
    
    print("\nReject Null Hypothesis")
    
    print(
        "Bike rentals significantly differ across seasons"
    )

else:
    
    print("\nFail to Reject Null Hypothesis")
    
    print(
        "Bike rentals do not significantly differ across seasons"
    )


# =========================================
# ANOVA TEST - WEATHER
# =========================================

print("\n================ ANOVA TEST - WEATHER ================\n")


weather_1 = df[
    df['weather'] == 1
]['count']


weather_2 = df[
    df['weather'] == 2
]['count']


weather_3 = df[
    df['weather'] == 3
]['count']


anova_weather_stat, anova_weather_p = f_oneway(
    weather_1,
    weather_2,
    weather_3
)


print("\nANOVA Statistic:")

print(anova_weather_stat)


print("\nANOVA P-Value:")

print(anova_weather_p)


if anova_weather_p < 0.05:
    
    print("\nReject Null Hypothesis")
    
    print(
        "Bike rentals significantly differ across weather conditions"
    )

else:
    
    print("\nFail to Reject Null Hypothesis")
    
    print(
        "Bike rentals do not significantly differ across weather conditions"
    )


# =========================================
# CHI-SQUARE TEST
# =========================================

print("\n================ CHI-SQUARE TEST ================\n")


contingency_table = pd.crosstab(
    df['season'],
    df['weather']
)


chi_stat, chi_p, dof, expected = chi2_contingency(
    contingency_table
)


print("\nChi-Square Statistic:")

print(chi_stat)


print("\nP-Value:")

print(chi_p)


if chi_p < 0.05:
    
    print("\nReject Null Hypothesis")
    
    print(
        "Weather condition is dependent on season"
    )

else:
    
    print("\nFail to Reject Null Hypothesis")
    
    print(
        "Weather condition is independent of season"
    )


# =========================================
# BUSINESS INSIGHTS
# =========================================

print("\n================ BUSINESS INSIGHTS ================\n")

print("""
1. Working days significantly impact bike rental demand.

2. Seasonal variation strongly affects electric cycle usage.

3. Weather conditions directly influence customer demand.

4. Peak rental demand occurs during office commuting hours.

5. Adverse weather conditions reduce bike rental activity.

6. Operational planning should consider seasonal demand fluctuations.

7. Yulu can optimize fleet allocation using demand forecasting patterns.
""")


# =========================================
# EXPORT CLEANED DATASET
# =========================================

df.to_csv(
    "../results/cleaned_yulu_data.csv",
    index=False
)


print("\n================ PIPELINE EXECUTED SUCCESSFULLY ================\n")