import torch


price = torch.tensor([500.0])

increase = price * 0.20
new_price = price * 1.20

print("Increase:", increase)
print("New price:", new_price)