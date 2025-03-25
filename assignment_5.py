import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


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

# Prepare the training data
X_train = train_data.drop(
    ["PassengerId", "Survived", "Name", "Ticket", "Cabin", "SibSp", "Parch", "FareBin"],
    axis=1,
)
y_train = train_data["Survived"]

# Prepare the test data
X_test = test_data.drop(
    ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch", "FareBin"], axis=1
)

# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Get number of features after preprocessing
n_features = X_train_preprocessed.shape[1]
print(f"Number of features after preprocessing: {n_features}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_preprocessed)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_preprocessed)


# Define the ANN model with 4 hidden layers
class ANN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(ANN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.hidden2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.hidden3 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.output_layer = nn.Linear(hidden_sizes[3], output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.sigmoid(self.output_layer(x))
        return x


# Function to train model and plot loss curve
def train_model(model, X_train, y_train, learning_rate, num_epochs=2500):
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Lists to store loss values for plotting
    losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store loss
        losses.append(loss.item())

        # Print progress
        if (epoch + 1) % 250 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(f"Loss vs. Epoch (Learning Rate: {learning_rate})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"training_graphs/loss_curve_lr_{learning_rate}.png")
    plt.show()

    return model, losses


# Define hidden layer sizes
hidden_sizes = [64, 32, 16, 8]

# Hyperparameters (learning rates)
learning_rates = [0.01, 0.001, 0.0001]
best_model = None
lowest_loss = float("inf")

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")

    # Initialize model
    model = ANN(input_size=n_features, hidden_sizes=hidden_sizes)

    # Train model
    trained_model, losses = train_model(model, X_train_tensor, y_train_tensor, lr)

    # Check if this model has lower final loss
    final_loss = losses[-1]
    if final_loss < lowest_loss:
        lowest_loss = final_loss
        best_model = trained_model
        best_lr = lr

    print(f"Final loss: {final_loss:.6f}")

print(f"\nBest model has learning rate {best_lr} with final loss: {lowest_loss:.6f}")

# Use the best model to make predictions
best_model.eval()
with torch.no_grad():
    predictions_tensor = best_model(X_test_tensor)
    # Convert probabilities to binary predictions
    predictions = (predictions_tensor > 0.5).int().numpy().flatten()

# Create submission file
submission = pd.DataFrame({"PassengerId": test_passenger_ids, "Survived": predictions})
submission.to_csv("kaggle_submission/ann_titanic_submission.csv", index=False)
print("\nSubmission file created: ann_titanic_submission.csv")

# Additional information for the report
print("\nModel Architecture:")
print(best_model)
print(f"\nBest Learning Rate: {best_lr}")
print(f"Final Loss: {lowest_loss:.6f}")
