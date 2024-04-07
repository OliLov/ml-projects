"""
This script implements a simple Convolutional Neural Network (CNN) for emotion
classification using PyTorch.

Usage:
    python cnn.py --train /path/to/trainset
    python cnn.py --test /path/to/testset

Arguments:
    --train (str): Path to the directory containing the training dataset.
    --test (str): Path to the directory containing the testing dataset.
    --lr (float, optional): Learning rate for the optimizer. Default is 0.001.
    --momentum (float, optional): Momentum for the SGD optimizer. Default is 0.9.
    --batch_size (int, optional): Batch size for training. Default is 32.
    --epochs (int, optional): Number of epochs for training. Default is 5.
    --save_path (str, optional): Path to save the trained model. Default is "model.pth".
    --predictions (str, optional): Path to save the predictions. Default is None.
"""

import argparse
import json

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class SimpleCNN(nn.Module):
    """Simple CNN"""

    def __init__(self):
        """Initalize Simple CNN."""
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 12 * 12, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=7)

    def forward(self, x):
        """Forward pass through the model.

        :param x: Input tensor.
        :return: Output tensor.
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = x.view(-1, 64 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(trainloader, lr=0.001, momentum=0.9, epochs=5, save_path="model.pth"):
    """Train the model.

    :param trainloader: DataLoader for training dataset.
    :param lr: Learning rate for the optimizer. Default is 0.001.
    :param momentum: Momentum for the SGD optimizer. Default is 0.9.
    :param epochs: Number of epochs for training. Default is 5.
    :param save_path: Path to save the trained model. Default is "model.pth".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the network, loss function, and optimizer
    net = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        with tqdm(
            total=len(trainloader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
        ) as pbar:
            for i, data in enumerate(trainloader):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    pbar.set_postfix(loss=running_loss / 100)
                    running_loss = 0.0
                pbar.update(1)

    print("Finished Training.")

    # Save the trained model
    torch.save(net.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def test_model(model, testloader, predictions_path=None):
    """Test the model.

    :param model: Trained model to be tested.
    :param testloader: DataLoader for testing dataset.
    :param predictions_path: Path to save the predictions. Default is None.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for image_path, expected_label, predicted_label in zip(
                testloader.dataset.samples, labels, predicted
            ):
                prediction = {
                    "image_name": image_path[0],
                    "expected_label": expected_label.item(),
                    "predicted_label": predicted_label.item(),
                }
                predictions.append(prediction)

    print("Accuracy of the network on the test images: %d %%" % (100 * correct / total))

    # Save predictions to a JSON file
    if predictions_path:
        with open(predictions_path, "w") as f:
            json.dump(predictions, f)
        print(f"Predictions saved to {predictions_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple CNN for emotion classification"
    )
    parser.add_argument(
        "--train", type=str, default=None, help="path to the train dataset"
    )
    parser.add_argument(
        "--test", type=str, default=None, help="path to the test dataset"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum for SGD optimizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of epochs for training"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="model.pth",
        help="path to save the trained model",
    )
    parser.add_argument(
        "--predictions", type=str, default=None, help="path to save the predictions"
    )
    args = parser.parse_args()

    if args.train:
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        trainset = ImageFolder(root=args.train, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=8
        )

        train_model(
            trainloader,
            lr=args.lr,
            momentum=args.momentum,
            epochs=args.epochs,
            save_path=args.save_path,
        )

    if args.test:
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        testset = ImageFolder(root=args.test, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=8
        )

        model = SimpleCNN()
        model.load_state_dict(torch.load(args.save_path))
        test_model(model, testloader, predictions_path=args.predictions)
