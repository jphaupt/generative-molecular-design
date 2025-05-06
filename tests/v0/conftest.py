from pytest import fixture
from mygenai.utils.transforms import CompleteGraph, SetTarget, PadToFixedSize, ExtractFeatures
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from mygenai.models.graphvae import GraphVAE
import torch

@fixture
def v0_transforms():
    return Compose([
        ExtractFeatures(),
        PadToFixedSize(),
        CompleteGraph(),
        SetTarget()
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

@fixture
def v0_dataloader(v0_qm9_dataset):
    """Create a dataloader for testing"""
    return DataLoader(v0_qm9_dataset, batch_size=4, shuffle=False)

@fixture
def v0_single_batch(v0_dataloader, device):
    """Get a single batch for testing"""
    return next(iter(v0_dataloader)).to(device)

@fixture
def v0_random_latent_vector(v0_single_batch, device):
    """Generate a random latent vector (z) for testing"""
    batch_size = v0_single_batch.num_graphs
    latent_dim = 32
    return torch.randn(batch_size, latent_dim).to(device)

@fixture
def water(v0_qm9_dataset):
    # with v0_transforms
    # x = [29, 5]
    # edge_index = [2, 812]
    # edge_attr = [812, 4]
    return v0_qm9_dataset[2]


@fixture
def ch3cn(v0_qm9_dataset):
    return v0_qm9_dataset[9]
