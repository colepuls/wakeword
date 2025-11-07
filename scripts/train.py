from dataset import train_feats, val_feats
from model import WakewordRNN
import torch.optim as optim, torch.nn as nn, torch, random

model = WakewordRNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

BATCH_SIZE = 32
EPOCHS = 5

for epoch in range(EPOCHS):
    # Train
    model.train()
    random.shuffle(train_feats)
    total_loss, total_count = 0.0, 0

    for i in range(0, len(train_feats), BATCH_SIZE):
        batch = train_feats[i:i+BATCH_SIZE]
        xs = torch.stack([x for x, _ in batch]) # (B, 40, 101)
        ys = torch.tensor([y for _, y in batch], dtype=torch.float32).unsqueeze(1) # (B, 1)

        # Forward pass
        pred = model(xs)
        loss = criterion(pred, ys)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xs.size(0)
        total_count += xs.size(0)

    avg_loss = total_loss / total_count

    # Validate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(val_feats), BATCH_SIZE):
            batch = val_feats[i:i+BATCH_SIZE]
            xs = torch.stack([x for x, _ in batch])
            ys = torch.tensor([y for _, y in batch], dtype=torch.int32)

            pred = model(xs).squeeze(1) # (B,)
            predicted = (pred > 0.5).int()
            correct += int((predicted == ys).sum().item())
            total += xs.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS} | loss={avg_loss:.4f} | val_acc={correct/total:.2%}")

# Save model!!! lets go please work
torch.save(model.state_dict(), "wakeword_model.pth")
print("Model saved as wakeword_model.pth\n")