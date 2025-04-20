# shared fixtures
import pytest
import torch
import numpy as np
from scipy.stats import ortho_group
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from mygenai.utils.transforms import CompleteGraph

@pytest.fixture
def random_orthogonal_matrix(dim=3):
  """Helper function to build a random orthogonal matrix of shape (dim, dim)
  """
  Q = torch.tensor(ortho_group.rvs(dim=dim)).float()
  return Q

@pytest.fixture
def qm9_dataset():
    """Load a small subset of QM9 dataset for testing"""
    dataset = QM9(root="./data/QM9", transform=CompleteGraph())
    # Only use first 100 examples to make tests faster
    dataset = dataset[:100]

    # Normalize targets to mean = 0 and std = 1 for the subset only
    mean = dataset.y.mean(dim=0, keepdim=True)  # Access the subset's y attribute
    std = dataset.y.std(dim=0, keepdim=True)
    dataset.y = (dataset.y - mean) / std

    return dataset

@pytest.fixture
def dataloader(qm9_dataset):
    """Create a dataloader for testing"""
    return DataLoader(qm9_dataset, batch_size=4, shuffle=False)

@pytest.fixture
def single_batch(dataloader):
    """Get a single batch for testing"""
    return next(iter(dataloader))

@pytest.fixture
def device():
    """Get available device (CPU or CUDA)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
