
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# =====================================================
# Part A: Understanding the Dataset
# =====================================================

# Q1. Dataset Overview

df = pd.read_csv("/Users/anshikakansal/Downloads/agriculture_yield_dataset.csv")

print("\nQ1 Dataset Overview")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn Names:")
print(df.columns)

print("\nFirst 10 Records:")
print(df.head(10))


# Q2. Data Types and Missing Values

print("\nQ2 Data Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())


# Q3. Descriptive Statistics

print("\nQ3 Descriptive Statistics")
print(df.describe())

highest_mean = df.mean(numeric_only=True).idxmax()
highest_std = df.std(numeric_only=True).idxmax()

print(f"\nFeature with Highest Mean: {highest_mean}")
print(f"Feature with Highest Standard Deviation: {highest_std}")


# =====================================================
# Part B: Exploratory Data Analysis (EDA)
# =====================================================

# Q4. Distribution Analysis

plt.hist(df["rainfall_mm"], bins=20)
plt.title("Rainfall Distribution")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Frequency")
plt.show()

plt.hist(df["temperature_c"], bins=20)
plt.title("Temperature Distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.show()

plt.hist(df["fertilizer_kg"], bins=20)
plt.title("Fertilizer Distribution")
plt.xlabel("Fertilizer (kg)")
plt.ylabel("Frequency")
plt.show()

plt.hist(df["yield_ton_per_hectare"], bins=20)
plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()


# Q5. Crop Type Analysis

print("\nQ5 Crop Type Counts")
print(df["crop_type"].value_counts())

sns.countplot(x="crop_type", data=df)
plt.title("Crop Type Count")
plt.show()

most_common_crop = df["crop_type"].value_counts().idxmax()
print(f"Most Common Crop: {most_common_crop}")


# Q6. Soil Type Analysis

print("\nQ6 Soil Type Counts")
print(df["soil_type"].value_counts())

sns.countplot(x="soil_type", data=df)
plt.title("Soil Type Count")
plt.show()

most_common_soil = df["soil_type"].value_counts().idxmax()
print(f"Most Common Soil Type: {most_common_soil}")


# Q7. Yield Distribution

plt.hist(df["yield_ton_per_hectare"], bins=20)
plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()


# Q8. Scatter Plot Analysis

plt.scatter(df["rainfall_mm"], df["yield_ton_per_hectare"])
plt.xlabel("Rainfall")
plt.ylabel("Yield")
plt.title("Rainfall vs Yield")
plt.show()

plt.scatter(df["fertilizer_kg"], df["yield_ton_per_hectare"])
plt.xlabel("Fertilizer")
plt.ylabel("Yield")
plt.title("Fertilizer vs Yield")
plt.show()


# Q9. Correlation Analysis

corr_matrix = df.corr(numeric_only=True)

print("\nQ9 Correlation Matrix")
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Heatmap")
plt.show()

yield_corr = corr_matrix["yield_ton_per_hectare"].sort_values(ascending=False)

print("\nCorrelation with Yield")
print(yield_corr)


# Q10. Group-Based Analysis

crop_yield = df.groupby("crop_type")["yield_ton_per_hectare"].mean()

print("\nAverage Yield by Crop Type")
print(crop_yield)

soil_yield = df.groupby("soil_type")["yield_ton_per_hectare"].mean()

print("\nAverage Yield by Soil Type")
print(soil_yield)

best_crop = crop_yield.idxmax()
best_soil = soil_yield.idxmax()

print(f"\nCrop with Highest Average Yield: {best_crop}")
print(f"Soil Type with Highest Average Yield: {best_soil}")


# =====================================================
# Part C: Data Preparation
# =====================================================

# Q11. Feature Encoding

categorical_columns = df.select_dtypes(include="object").columns

print("\nCategorical Columns")
print(categorical_columns)

df_encoded = pd.get_dummies(df, columns=categorical_columns)

print("\nFirst 5 Rows After Encoding")
print(df_encoded.head())


# Q12. Feature Selection

X = df_encoded.drop("yield_ton_per_hectare", axis=1)
y = df_encoded["yield_ton_per_hectare"]

print("\nTarget Variable:")
print("yield_ton_per_hectare")


# =====================================================
# Part D: Machine Learning
# =====================================================

# Q13. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nQ13 Shapes")

print(f"X_train Shape: {X_train.shape}")
print(f"X_test Shape: {X_test.shape}")
print(f"y_train Shape: {y_train.shape}")
print(f"y_test Shape: {y_test.shape}")


# Q14. Linear Regression Model

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Coefficients")
print(model.coef_)

print(f"\nIntercept: {model.intercept_}")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients")
print(coefficients)

highest_feature = coefficients.sort_values(
    by="Coefficient",
    ascending=False
).iloc[0]

print(
    f"\nFeature with Highest Positive Coefficient: "
    f"{highest_feature['Feature']}"
)