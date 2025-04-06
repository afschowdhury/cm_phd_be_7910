import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


def create_mnist_csv(train=True, output_dir="./mnist_data"):
    """
    Create a CSV file with image names and labels for MNIST dataset
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "train"))
        os.makedirs(os.path.join(output_dir, "test"))

    folder = "train" if train else "test"

    mnist_dataset = datasets.MNIST(
        root="./data", train=train, download=True, transform=transforms.ToTensor()
    )

    image_names = []
    labels = []

    for i, (image, label) in enumerate(mnist_dataset):

        image_pil = transforms.ToPILImage()(image)
        image_name = f"{i:05d}.png"
        image_path = os.path.join(output_dir, folder, image_name)
        image_pil.save(image_path)

        image_names.append(image_name)
        labels.append(label)

    df = pd.DataFrame({"image_name": image_names, "label": labels})
    csv_path = os.path.join(output_dir, f"{folder}.csv")
    df.to_csv(csv_path, index=False)

    print(f"Created {csv_path} with {len(df)} records")
    return csv_path


class MNISTDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with image names and labels
            img_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("L")  # Convert to grayscale
        label = self.data_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Input size: 28x28 = 784
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10)  # 10 classes for digits 0-9
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        x = x.view(-1, 784)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.relu(self.fc4(x))
        x = self.dropout(x)

        x = self.fc5(x)
        return x


def train_model(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def test_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = 100.0 * correct / total

    return test_loss, test_acc


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = "./mnist_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.makedirs(os.path.join(data_dir, "train"))
        os.makedirs(os.path.join(data_dir, "test"))

    train_csv_path = os.path.join(data_dir, "train.csv")
    test_csv_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        print("Creating CSV files for train and test data...")
        train_csv_path = create_mnist_csv(train=True, output_dir=data_dir)
        test_csv_path = create_mnist_csv(train=False, output_dir=data_dir)
    else:
        print("CSV files already exist. Skipping creation.")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    train_dataset = MNISTDataset(
        csv_file=train_csv_path,
        img_dir=os.path.join(data_dir, "train"),
        transform=transform,
    )

    test_dataset = MNISTDataset(
        csv_file=test_csv_path,
        img_dir=os.path.join(data_dir, "test"),
        transform=transform,
    )

    batch_size = 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    learning_rates = [0.01, 0.001, 0.0001]
    num_epochs = 15

    results = {}

    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")

        model = MNISTNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_test_acc = 0.0
        best_model_path = None

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for epoch in range(num_epochs):

            train_loss, train_acc = train_model(
                model, train_loader, optimizer, criterion, device
            )

            test_loss, test_acc = test_model(model, test_loader, criterion, device)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            print(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
            )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_path = (
                    f"./mnist_model_lr_{lr:.6f}_acc_{best_test_acc:.2f}.pth"
                )
                torch.save(model.state_dict(), best_model_path)
                print(f"Model saved to {best_model_path}")

        results[lr] = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "test_losses": test_losses,
            "test_accs": test_accs,
            "best_test_acc": best_test_acc,
            "best_model_path": best_model_path,
        }

    plt.figure(figsize=(15, 10))

    for i, lr in enumerate(learning_rates):
        plt.subplot(2, 3, i + 1)
        plt.plot(results[lr]["train_losses"], label="Train Loss")
        plt.plot(results[lr]["test_losses"], label="Test Loss")
        plt.title(f"Loss (LR={lr})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(2, 3, i + 4)
        plt.plot(results[lr]["train_accs"], label="Train Acc")
        plt.plot(results[lr]["test_accs"], label="Test Acc")
        plt.title(f"Accuracy (LR={lr})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

    plt.tight_layout()
    plt.savefig("mnist_training_results.png")

    print("\nBest test accuracies:")
    for lr in learning_rates:
        print(f"Learning rate {lr}: {results[lr]['best_test_acc']:.2f}%")
        print(f"Best model saved at: {results[lr]['best_model_path']}")

    best_lr = max(learning_rates, key=lambda lr: results[lr]["best_test_acc"])
    best_overall_model_path = results[best_lr]["best_model_path"]
    best_overall_acc = results[best_lr]["best_test_acc"]

    print(f"\nBest overall model:")
    print(f"Learning rate: {best_lr}")
    print(f"Test accuracy: {best_overall_acc:.2f}%")
    print(f"Model path: {best_overall_model_path}")

    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(best_overall_model_path))
    model.eval()

    _, test_acc = test_model(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Loaded best model accuracy: {test_acc:.2f}%")


def load_and_predict(model_path, test_loader, device=None):
    """
    Load a saved model and make predictions on the test set
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Model loaded from {model_path}")
    print(f"Test accuracy: {accuracy:.2f}%")

    return accuracy


if __name__ == "__main__":
    main()
