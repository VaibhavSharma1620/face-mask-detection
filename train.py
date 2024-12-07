
import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
NUM_CLASSES = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS =50
EARLY_STOPPING_PATIENCE = 10

# Custom Dataset
class MaskDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for xml_file in os.listdir(self.annotation_dir):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(self.annotation_dir, xml_file)
                img_path = os.path.join(self.image_dir, xml_file[:-4] + '.png')

                tree = ET.parse(xml_path)
                root = tree.getroot()

                objects = root.findall('object')
                for obj in objects:
                    name = obj.find('name').text
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)

                    samples.append((img_path, (xmin, ymin, xmax, ymax), name))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        # Crop the image to the bounding box
        image = image.crop(bbox)

        if self.transform:
            image = self.transform(image)

        # Convert label to numeric
        label_map = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}
        label = label_map[label]

        return image, label

# Model Definition (ResNet18 with modified final layer)
def create_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_acc = 0.0
    best_model_wts = model.state_dict()
    counter = 0

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
                scheduler.step()
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = model.state_dict()
                counter = 0
            elif phase == 'val':
                counter += 1

        if counter >= patience:
            print("Early stopping")
            break

    print(f'Best val Acc: {best_val_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model, train_losses, train_accs, val_losses, val_accs

# Inference function
def infer(model, image_path, bbox):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = image.crop(bbox)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    label_map = {0: 'with_mask', 1: 'without_mask', 2: 'mask_weared_incorrect'}
    return label_map[preds.item()]

# Main execution
def main():
    # Set up data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = MaskDataset('/content/drive/MyDrive/c/archive/images', '/content/drive/MyDrive/c/archive/annotations', transform=transform)
    train_val_data, test_data = train_test_split(full_dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Create model and set up training
    model = create_model()
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train model
    model, train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, EARLY_STOPPING_PATIENCE
    )

    # Save model
    torch.save(model.state_dict(), 'mask_detection_model.pth')

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss')
    plt.legend()

    # plt.subplot(122)
    # plt.plot(train_accs, label='Train')
    # plt.plot(val_accs, label='Validation')
    # plt.title('Accuracy')
    # plt.legend()

    plt.savefig('training_curves.png')
    plt.close()


if __name__ == '__main__':
    main()