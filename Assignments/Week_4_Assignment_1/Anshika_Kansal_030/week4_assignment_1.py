import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================================================
# Part A: Initial Analysis
# =====================================================

# Q1. Load Dataset

df = pd.read_csv("/Users/anshikakansal/Downloads/Loan prediction.csv")

print("\nQ1. First 10 Records")
print(df.head(10))

print("\nFeatures")
print(df.columns[:-1])

print(f"\nTarget Variable: {df.columns[-1]}")

print(f"\nRows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())


# =====================================================
# Part B: Data Preprocessing
# =====================================================

# Q2

num_cols = df.select_dtypes(include=["int64","float64"]).columns
cat_cols = df.select_dtypes(include="object").columns

num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")

df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("\nData preprocessing completed.")


# =====================================================
# Part C: Exploratory Data Analysis
# =====================================================

# Q3

plt.figure(figsize=(6,4))
sns.countplot(x="Loan_Status", data=df)
plt.title("Loan Approval Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=df)
plt.title("Applicant Income vs Loan Approval")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Credit_History", hue="Loan_Status", data=df)
plt.title("Credit History vs Loan Approval")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Education", hue="Loan_Status", data=df)
plt.title("Education vs Loan Approval")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Property_Area", hue="Loan_Status", data=df)
plt.title("Property Area vs Loan Approval")
plt.show()


# =====================================================
# Part D: Machine Learning
# =====================================================

# Q4

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nX_train Shape: {X_train.shape}")
print(f"X_test Shape: {X_test.shape}")
print(f"y_train Shape: {y_train.shape}")
print(f"y_test Shape: {y_test.shape}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    results.append([name, accuracy, precision, recall, f1])

results_df = pd.DataFrame(
    results,
    columns=["Model","Accuracy","Precision","Recall","F1 Score"]
)

print("\nModel Comparison")
print(results_df)


# =====================================================
# Part E: Model Evaluation
# =====================================================

# Q5

best_model = results_df.loc[results_df["Accuracy"].idxmax()]

print("\nBest Performing Model")
print(best_model)


# =====================================================
# Part F: Stratified 5-Fold Cross Validation
# =====================================================

# Q6

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nCross Validation Results")

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring="accuracy"
    )

    print(f"\n{name}")
    print(f"Fold Accuracy: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")


# =====================================================
# Part G: Hyperparameter Tuning
# =====================================================

# Q7

params = {

    "n_estimators":[50,100,200],
    "max_depth":[3,5,10],
    "min_samples_split":[2,5,10]

}

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    rf,
    params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print("\nBest Parameters")
print(grid.best_params_)

print(f"\nBest Cross Validation Score: {grid.best_score_:.4f}")

best_rf = grid.best_estimator_

prediction = best_rf.predict(X_test)

print(f"Test Accuracy After Tuning: {accuracy_score(y_test,prediction):.4f}")


# =====================================================
# Part H: Bias-Variance Tradeoff
# =====================================================

# Q8

depths = [2,5,15]

bias_results = []

for depth in depths:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train,y_train)

    train_accuracy = accuracy_score(
        y_train,
        tree.predict(X_train)
    )

    test_accuracy = accuracy_score(
        y_test,
        tree.predict(X_test)
    )

    bias_results.append(
        [depth,train_accuracy,test_accuracy]
    )

bias_df = pd.DataFrame(
    bias_results,
    columns=[
        "Max Depth",
        "Training Accuracy",
        "Testing Accuracy"
    ]
)

print("\nBias Variance Comparison")
print(bias_df)