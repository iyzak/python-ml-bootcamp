
import torch

# 1. Prepare the training examples
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. Create a model with one input and one output
model = torch.nn.Linear(1, 1)

# 3. Choose how to measure errors
loss_fn = torch.nn.MSELoss()

# 4. Choose how the model will learn
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 5. Train the model
for epoch in range(1000):
    predictions = model(x_train)
    loss = loss_fn(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 6. Display what the model learned
print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())
print("Final loss:", loss.item())

# 7. Predict the output for a new input
new_input = torch.tensor([[5.0]])
prediction = model(new_input)

print("Prediction for x = 5:", prediction.item())
