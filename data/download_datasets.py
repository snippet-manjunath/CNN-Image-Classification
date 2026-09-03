"""Download MNIST and CIFAR-10 datasets."""

import argparse
import os
import torch
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

def download_mnist(data_path="./data/mnist"):
    """Download MNIST dataset."""
    print("Downloading MNIST dataset...")
    os.makedirs(data_path, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    print("✓ MNIST download complete")

def download_cifar10(data_path="./data/cifar10"):
    """Download CIFAR-10 dataset."""
    print("Downloading CIFAR-10 dataset...")
    os.makedirs(data_path, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    print("✓ CIFAR-10 download complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--mnist", action="store_true", help="Download MNIST")
    parser.add_argument("--cifar10", action="store_true", help="Download CIFAR-10")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    args = parser.parse_args()
    
    if args.all or args.mnist:
        download_mnist()
    if args.all or args.cifar10:
        download_cifar10()
    
    if not (args.all or args.mnist or args.cifar10):
        print("Usage: python download_datasets.py [--mnist] [--cifar10] [--all]")
