import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 1. Load the data
train_data = pd.read_csv("datasets/titanic/train.csv")
test_data = pd.read_csv("datasets/titanic/test.csv")

# Save passenger IDs for submission file
test_passenger_ids = test_data["PassengerId"]

# Display basic info about the data
print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# 2. Data Exploration
print("\nTraining data info:")
print(train_data.info())
print("\nMissing values in training data:")
print(train_data.isnull().sum())

print("\nTest data info:")
print(test_data.info())
print("\nMissing values in test data:")
print(test_data.isnull().sum())

# 3. Feature Engineering


# 3.1 Create new features
def add_features(df):
    """Add additional features to the dataframe"""
    # Family size feature
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Is alone feature
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Extract titles from names
    df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())

    # Group uncommon titles
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Mlle": "Miss",
        "Countess": "Rare",
        "Ms": "Miss",
        "Lady": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
        "Mme": "Mrs",
        "Capt": "Rare",
        "Sir": "Rare",
    }
    df["Title"] = df["Title"].map(lambda x: title_mapping.get(x, "Rare"))

    # Bin fare values
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False, duplicates="drop")

    # Create deck feature from cabin
    df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) else "U")

    # Group rare decks
    df["Deck"] = df["Deck"].replace(["T", "G"], "U")

    return df


# Apply feature engineering to both datasets
train_data = add_features(train_data)
test_data = add_features(test_data)

# 3.2 Correlation Analysis
# Calculate correlation matrix for training data
correlation_matrix = train_data[
    ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
].corr()
print("\nCorrelation matrix:")
print(correlation_matrix["Survived"].sort_values(ascending=False))

# 4. Feature Selection
# Based on correlation analysis and domain knowledge

# Numeric features requiring scaling
numeric_features = ["Age", "Fare", "FamilySize"]

# Categorical features requiring one-hot encoding
categorical_features = ["Pclass", "Sex", "Embarked", "Title", "Deck", "IsAlone"]

# Define feature transformers
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 5. Build the Model Pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# Prepare the training data
X_train = train_data.drop(
    ["PassengerId", "Survived", "Name", "Ticket", "Cabin", "SibSp", "Parch", "FareBin"],
    axis=1,
)
y_train = train_data["Survived"]

# 6. Model Tuning with Grid Search
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("\nBest parameters:")
print(best_params)
print(f"\nBest CV accuracy: {grid_search.best_score_:.4f}")

# 7. Feature Importance Analysis
feature_names = numeric_features + list(
    best_model.named_steps["preprocessor"]
    .transformers_[1][1]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features)
)

importances = best_model.named_steps["classifier"].feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature ranking:")
for i, idx in enumerate(indices[:10]):
    if i < len(feature_names):
        print(f"{i+1}. {feature_names[idx]} ({importances[idx]:.4f})")

# 8. Generate Predictions on Test Set
X_test = test_data.drop(
    ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch", "FareBin"], axis=1
)
predictions = best_model.predict(X_test)

# 9. Create Submission File
submission = pd.DataFrame({"PassengerId": test_passenger_ids, "Survived": predictions})

submission.to_csv("titanic_submission.csv", index=False)
print("\nSubmission file created: titanic_submission.csv")
