import torch

def train_epoch(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        logits, mu, logvar, property_pred = model(batch)

        # Calculate loss
        loss = model.loss_function(logits, batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = batch.to(device)

            # Forward pass
            logits, mu, logvar, property_pred = model(batch)

            # Calculate loss
            loss = model.loss_function(logits, batch)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_model(model, train_loader, val_loader, device, n_epochs=100, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    epochs_without_improvement = 0  # Counter for early stopping

    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, optimizer, train_loader, device)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Print progress
        print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1

        # Stop training if validation loss hasn't improved for `patience` epochs
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print("Training finished")
