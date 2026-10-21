
import torch

# Training examples
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Separate validation examples
x_val = torch.tensor([[5.0], [6.0]])
y_val = torch.tensor([[10.0], [12.0]])

# Create the model and training tools
model = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model and monitor both losses
for epoch in range(1001):
    # Learn from training data
    train_predictions = model(x_train)
    train_loss = loss_fn(train_predictions, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Check performance on unseen validation data
    if epoch % 100 == 0:
        with torch.no_grad():
            val_predictions = model(x_val)
            val_loss = loss_fn(val_predictions, y_val)

        print(
            f"Epoch {epoch:4d} | "
            f"Training loss: {train_loss.item():.4f} | "
            f"Validation loss: {val_loss.item():.4f}"
        )

# Test a new input
with torch.no_grad():
    test_input = torch.tensor([[7.0]])
    test_prediction = model(test_input)

print("Prediction for x=7:", test_prediction.item())
