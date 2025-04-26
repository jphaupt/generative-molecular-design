# we want to make sure that the model can faithfully reconstruct a single molecule
# to test this, we will overfit to a single molecule and check reconstruction metrics

import copy, torch

def test_overfit_single_molecule(v0_model, v0_single_batch, device):
    """Test that the model can overfit to a single molecule."""
    assert False, "Placeholder: first playing with this in a notebook"
    # TODO test on a specific molecule so you can test more carefully?
    model = v0_model.deepcopy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    # Overfit to a single batch
    # TODO use training functions already defined!
    for epoch in range(100):  # Overfitting for 100 epochs
        optimizer.zero_grad()
        logits, mu, logvar, property_pred = model(v0_single_batch)
        loss = model.loss_function(logits, v0_single_batch)
        loss.backward()
        optimizer.step()

    # Check that the loss is low
    assert loss.item() < 0.1, f"Loss is too high: {loss.item()}"
