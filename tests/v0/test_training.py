import pytest
from mygenai.training.training import train_epoch, validate, train_model
import copy
import torch
import numpy as np
import random

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set seed for reproducibility
set_seed()
def test_train_epoch(v0_model, v0_dataloader, device):
    """Test that a single training epoch runs without errors and returns a valid loss."""
    model = copy.deepcopy(v0_model) # copy the model to avoid modifying the original
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run a single epoch
    loss = train_epoch(model, optimizer, v0_dataloader, device)

    # Check loss validity
    assert isinstance(loss, float), f"Expected float loss value, got {type(loss)}"
    assert not np.isnan(loss), "Training loss is NaN"
    assert not np.isinf(loss), "Training loss is infinite"
    assert loss > 0, f"Expected positive loss, got {loss}"

def test_validate(v0_model, v0_dataloader, device):
    """Test that validation runs without errors and returns a valid loss."""
    model = copy.deepcopy(v0_model)
    loss = validate(model, v0_dataloader, device)

    # Check loss validity
    assert isinstance(loss, float), f"Expected float loss value, got {type(loss)}"
    assert not np.isnan(loss), "Validation loss is NaN"
    assert not np.isinf(loss), "Validation loss is infinite"
    assert loss > 0, f"Expected positive loss, got {loss}"

def test_train_model_runs(v0_model, v0_dataloader, device):
    """Test that the full training loop runs for a few epochs without errors."""
    model = copy.deepcopy(v0_model)

    # Run training for just 2 epochs
    train_model(model, v0_dataloader, v0_dataloader, device, n_epochs=2, patience=5)

    # If we got here without errors, the test passed
    assert True, "Training completed successfully"

def test_loss_decreases(v0_model, v0_dataloader, device):
    """Test that loss generally decreases during training."""
    model = copy.deepcopy(v0_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Get initial loss
    initial_loss = train_epoch(model, optimizer, v0_dataloader, device)

    # Train for a few more epochs
    for _ in range(5):
        current_loss = train_epoch(model, optimizer, v0_dataloader, device)

    # Check if loss decreased
    assert current_loss < initial_loss, f"Loss didn't decrease: {initial_loss} -> {current_loss}"

def test_early_stopping(v0_model, v0_dataloader, device, monkeypatch):
    """Test that early stopping works correctly"""
    assert False, "Placeholder"
