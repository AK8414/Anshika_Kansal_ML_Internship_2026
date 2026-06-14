
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- Part A: Dataset Understanding ----------------

# Q1. Load dataset and display first five records

df = pd.read_csv("/Users/anshikakansal/Downloads/Netflix_Dataset.csv")
print("First 5 Records:")
print(df.head())

# Q2. Number of rows and columns
print("\nRows and Columns:")
print(df.shape)

# Q3. Display column names
print("\nColumn Names:")
print(df.columns)

# Q4. Numerical and Categorical Features
print("\nNumerical Features:")
print(df.select_dtypes(include=['int64', 'float64']).columns)

print("\nCategorical Features:")
print(df.select_dtypes(include=['object']).columns)

# Q5. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# ---------------- Part B: Exploratory Data Analysis ----------------

# Q6. Average age of users
print("\nAverage Age:", df["Age"].mean())

# Q7. Average watch hours per week
print("Average Watch Hours Per Week:", df["WatchHoursPerWeek"].mean())

# Q8. Average monthly spending
print("Average Monthly Spending:", df["MonthlySpend"].mean())

# Q9. Count users in each subscription category
print("\nSubscription Counts:")
print(df["SubscriptionType"].value_counts())

# Q10. Percentage of users who renewed subscriptions
renewed_percentage = (df["SubscriptionRenewed"] == "Yes").mean() * 100
print("\nRenewed Percentage:", renewed_percentage)

# ---------------- Part C: Data Preparation ----------------

# Q11. Convert categorical features into numerical form
le = LabelEncoder()

for col in ["Gender", "SubscriptionType", "FavoriteGenre", "SubscriptionRenewed"]:
    df[col] = le.fit_transform(df[col])

# Q12. Define feature set and target variable
X = df.drop("SubscriptionRenewed", axis=1)
y = df["SubscriptionRenewed"]

# Q13. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# ---------------- Part D: Decision Tree Classification----------------

# Q14. Train Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Q15. Evaluate Decision Tree accuracy
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:", dt_accuracy)

# Q16. Confusion Matrix
cm = confusion_matrix(y_test, dt_pred)
print("\nConfusion Matrix:")
print(cm)

# ----------------Part E: K-Nearest Neighbors (KNN)----------------

# Q17. Train KNN Classifier (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Q18. Compare KNN Accuracy
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

print("\nKNN Accuracy:", knn_accuracy)

# ----------------Part F: Linear Regression----------------

# Q19. Linear Regression for Monthly Spending
X_reg = df.drop("MonthlySpend", axis=1)
y_reg = df["MonthlySpend"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)

# Q20. Predict Monthly Spending for a New User
new_user = X_reg.iloc[[0]]
prediction = lr.predict(new_user)

print("\nPredicted Monthly Spending:", prediction[0])

# -----------------------------
# Business Reflection Answers
# -----------------------------

print("\nBusiness Reflection Answers")

print("\n1. Factors influencing subscription renewal:")
print("MonthlySpend, Age, WatchHoursPerWeek, and AdClicks appear to have the greatest impact.")

print("\n2. Why is subscription renewal a classification problem?")
print("Because the outcome is categorical: Yes or No.")

print("\n3. Why is monthly spending a regression problem?")
print("Because monthly spending is a continuous numerical value.")

print("\n4. Which algorithm performed better?")
print("KNN performed slightly better than Decision Tree based on accuracy.")

print("\n5. How can Netflix use these predictions?")
print("Netflix can identify users likely to lea"
      "ve, provide personalized recommendations, offer discounts, and improve customer retention.")

