import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from model import GenderModel
from time import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

# --- data loading ---

data_root_dir = './data'


transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    train_dataset = torchvision.datasets.ImageFolder(root=f'{data_root_dir}/train', transform=transform)
    valid_dataset = torchvision.datasets.ImageFolder(root=f'{data_root_dir}/valid', transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=f'{data_root_dir}/test', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Loaded {len(train_dataset)} training images.")
    print(f"Loaded {len(valid_dataset)} validation images.")
    print(f"Loaded {len(test_dataset)} test images.")
    print(f"Classes: {train_dataset.classes}")

except Exception as e:
    print(f"Error loading datasets. Please ensure '{data_root_dir}' exists and has the correct subdirectory structure:")
    print(f"Error: {e}")

for images, labels in train_loader:
    print(f"\nShape of one batch of images: {images.shape}")
    print(f"Shape of one batch of labels: {labels.shape}")
    break

# --- data loading finished ---



model = GenderModel().to(device)
print("\nModel Architecture:")
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- training ---

num_epochs = 10
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

t1 = time()
print("\nStarting Training...")
for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    
    model.eval()
    valid_loss = 0.0
    correct_valid = 0
    total_valid = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_valid += labels.size(0)
            correct_valid += (predicted == labels).sum().item()

    avg_valid_loss = valid_loss / len(valid_loader)
    valid_accuracy = 100 * correct_valid / total_valid
    valid_losses.append(avg_valid_loss)
    valid_accuracies.append(valid_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%")

t2 = time()
print(f"Training Finished After {t2-t1} seconds!")

# --- training finished ---


# --- evaluating ---

print("\nEvaluating Model on Test Set...")
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

accuracy_test = 100 * correct_test / total_test
print(f"Accuracy of the network on the {total_test} test images: {accuracy_test:.2f}%")


# --- plot the resluts ---

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(valid_accuracies, label='Valid Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

a = input("Do you want to save the model? y/n")
if a == "y":
    path = 'models/model.pth'
    torch.save(model.state_dict(), path)
    print(f"\nModel weights saved to {path}")