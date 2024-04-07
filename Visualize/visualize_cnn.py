"""
This script provides visualizations for a trained Simple CNN model using PyTorch.

Usage:
    python visualize.py summary
    python visualize.py torchviz
    python visualize.py matplotlib

Arguments:
    summary: Visualize model summary.
    torchviz: Visualize model graph using torchviz.
    matplotlib: Visualize model filters using matplotlib.
"""

import argparse

import matplotlib.pyplot as plt
import torch
from torchinfo import summary
from torchviz import make_dot

from PyTorch.cnn import SimpleCNN


def visualize_summary(model, device):
    """Visualize model summary.

    :param model: Trained model to visualize.
    :param device: Device to run the model on.
    """
    summary(model.to(device), input_size=(1, 48, 48))


def visualize_torchviz(model, device):
    """Visualize model graph using torchviz.

    :param model: Trained model to visualize.
    :param device: Device to run the model on.
    """
    input = torch.randn(1, 1, 48, 48).to(device)
    output = model(input)
    graph = make_dot(output, params=dict(model.named_parameters()))
    graph.render("SimpleCNN", format="png", cleanup=True)


def visualize_matplotlib(model):
    """Visualize model filters using matplotlib.

    :param model: Trained model to visualize.
    """
    filters = model.conv1.weight.detach().numpy()
    for i in range(filters.shape[0]):
        plt.imshow(filters[i, 0], cmap="gray")
        plt.axis("off")
        plt.savefig(f"filter_{i}.png", bbox_inches="tight", pad_inches=0)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Neural Network Visualizer")
    parser.add_argument(
        "visualization",
        choices=["summary", "torchviz", "matplotlib", "tensorboard"],
        help="Type of visualization (summary, torchviz, matplotlib, tensorboard)",
    )
    args = parser.parse_args()

    # Load the model
    model = SimpleCNN()
    model.load_state_dict(torch.load("model.pth"))

    # Perform the specified visualization
    if args.visualization == "summary":
        visualize_summary(model)
    elif args.visualization == "torchviz":
        visualize_torchviz(model)
    elif args.visualization == "matplotlib":
        visualize_matplotlib(model)
