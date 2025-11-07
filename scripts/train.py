from dataset import train_feats
from model import WakewordRNN
import torch.optim as optim, torch.nn as nn, torch

model = WakewordRNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):
    total_loss = 0.0
    for x, y in train_feats[:32]:
        x = x.unsqueeze(0) # add batch dim
        y = torch.tensor([[y]], dtype=torch.float32)

        # Forward pass
        pred = model(x)
        loss = criterion(pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} | Avg loss: {total_loss / 32:.4f}")