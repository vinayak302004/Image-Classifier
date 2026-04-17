import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from PIL import Image
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_data = Subset(train_data, range(2000))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Model
model = models.resnet18(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(model.fc.in_features, 10)

# Train last layer + layer4
for param in model.layer4.parameters():
    param.requires_grad = True

model = model.to(device)

optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3}
])

loss_fn = torch.nn.CrossEntropyLoss()

# Training
model.train()
for epoch in range(5):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved!")