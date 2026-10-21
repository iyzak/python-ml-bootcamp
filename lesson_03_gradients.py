import torch

# Step 1: Create the input and the target answer
x = torch.tensor(3.0)
target = torch.tensor(12.0)

# Step 2: Create a weight that PyTorch can learn
weight = torch.tensor(2.0, requires_grad=True)

# Step 3: Make a prediction
prediction = weight * x

# Step 4: Calculate the error (loss)
loss = (prediction - target) ** 2

# Step 5: Calculate the gradient
loss.backward()

# Step 6: Display the results
print("Prediction:", prediction.item())
print("Target:", target.item())
print("Loss:", loss.item())
print("Gradient:", weight.grad.item())
