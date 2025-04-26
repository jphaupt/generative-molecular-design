import pytest
from pytest import fixture
from mygenai.utils.transforms import CompleteGraph, SetTarget, PadToFixedSize, ExtractFeatures
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from mygenai.models.graphvae import GraphVAE

@fixture
def v0_transforms():
    return Compose([
        ExtractFeatures(),
        PadToFixedSize(),
        CompleteGraph(),
        SetTarget()
        # NormalizeTarget()
    ])

@fixture
def v0_model(device):
    """v0 model with GCNConv encoder and edge existence decoder"""
    return GraphVAE().to(device)

@fixture
def v0_qm9_dataset(v0_transforms):
    """Load a small subset of QM9 dataset for testing"""
    dataset = QM9(root="./data/QM9", transform=v0_transforms)
    # Only use first 100 examples to make tests faster
    dataset = dataset[:100]
    mean = dataset.y.mean(dim=0, keepdim=True)
    std = dataset.y.std(dim=0, keepdim=True)
    dataset.y = (dataset.y - mean) / std
    return dataset

@pytest.fixture
def v0_dataloader(v0_qm9_dataset):
    """Create a dataloader for testing"""
    return DataLoader(v0_qm9_dataset, batch_size=4, shuffle=False)

@pytest.fixture
def v0_single_batch(v0_dataloader, device):
    """Get a single batch for testing"""
    return next(iter(v0_dataloader)).to(device)
