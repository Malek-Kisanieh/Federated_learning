"""flower2025: A Flower / PyTorch app.

Mhd Malek Kisanieh
Samir Akhalil"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)
import numpy as np

import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


fds = None
fds_type = None

pytorch_transforms = Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, use_iid: bool = True):
    """Load partition CIFAR10 data."""
    global fds, fds_type
    
    # Reset cache if switching partitioner type
    current_type = "iid" if use_iid else "non_iid"
    if fds is not None and fds_type != current_type:
        fds = None
        fds_type = None
    
    if fds is None:
        if use_iid:
            partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            # Non-IID with Dirichlet partitioner (alpha=0.5)
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=0.5,
                min_partition_size=10,
            )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
        fds_type = current_type
    
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device, is_attacker=False, attack_type="label_flipping"):
    """Train the model on the training set."""
    net.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)


    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            # Apply label flipping attack if this is an attacker
            if is_attacker:
                if attack_type == "label_flipping":
                    # Random label flipping
                    labels = torch.randint(0, 10, labels.shape, device=device)
                elif attack_type == "targeted_label_flipping":
                    # Shift labels by 1
                    labels = (labels + 1) % 10
            
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set and compute additional metrics."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    all_probs = []

    loss, correct = 0.0, 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            loss += criterion(outputs, labels).item()
            correct += (preds == labels).sum().item()

    # Calculate metrics
    accuracy = correct / len(testloader.dataset)
    avg_loss = loss / len(testloader)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    # ROC-AUC (multi-class)
    try:
        roc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except ValueError:
        roc = None

    return avg_loss, accuracy, kappa, f1, roc
