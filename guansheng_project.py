# %%

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import PIL.Image as Image

data_path = '/home/gs/Desktop/SS24 v5-6-neural_networks_and_deep_learning/data/Brain_Tumor_Classification_MRI/'

# %%
############################
# Dataset
############################

df = pd.DataFrame()

for tumor_type in os.listdir(data_path + "Training"):
    print(tumor_type)
    for img in os.listdir(data_path + "Training/" + tumor_type):

        df = df._append(
            {"image": img, "tumor_type": tumor_type, "data_set": "Training"},
            ignore_index=True,
        )

for tumor_type in os.listdir(data_path + "Testing"):
    print(tumor_type)
    for img in os.listdir(data_path + "Testing/" + tumor_type):
        df = df._append(
            {"image": img, "tumor_type": tumor_type, "data_set": "Testing"},
            ignore_index=True,
        )

df.head()


# %%
# transfer tumor type to label
def tumer_type_to_label(tumor_type):
    if tumor_type == "glioma_tumor":
        return 0
    elif tumor_type == "meningioma_tumor":
        return 1
    elif tumor_type == "no_tumor":
        return 2
    elif tumor_type == "pituitary_tumor":
        return 3

def label_to_tumer_type(label):
    if label == 0:
        return "glioma_tumor"
    elif label == 1:
        return "meningioma_tumor"
    elif label == 2:
        return "no_tumor"
    elif label == 3:
        return "pituitary_tumor"

# %%
df["label"] = df["tumor_type"].apply(tumer_type_to_label)

# %%
df["image_path"] = (
    data_path + df["data_set"] + "/" + df["tumor_type"] + "/" + df["image"]
)

# %%

# Visualize the images in the dataset
def image_plots(df, indices):
    fig, axs = plt.subplots(1, 4, figsize=(10, 10))

    for i, index in enumerate(indices):
        image = Image.open(df.iloc[index]["image_path"])

        ax = axs[i]
        ax.imshow(image)
        ax.axis("off")

        tumor = df.iloc[index]["tumor_type"]
        ax.set_title(f"Index: {index}\n{tumor}")

    plt.tight_layout()
    plt.show()


# %%

# np.random.seed(42)
indices = np.random.randint(0, len(df), 4)
image_plots(df, indices)

# %%

# Class for the dataset
class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_path"]
        image = Image.open(img_name)

        label = self.df.iloc[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

# %%
# df_train = df[df["data_set"] == "Training"]
# df_test = df[df["data_set"] == "Testing"]

df_train = df.sample(frac=0.8)
df_test = df.drop(df_train.index)


# %%
# Plot the distribution of the dataset
plt.title("Dataset Distribution")
plt.bar(df_train["tumor_type"].value_counts().index, df_train["tumor_type"].value_counts().values)
plt.bar(df_test["tumor_type"].value_counts().index, df_test["tumor_type"].value_counts().values)
plt.legend(["Train", "Test"])


# %%
"""
def calculate_mean_std(dataloader):
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std


# Calculate mean and standard deviation
# mean, std = calculate_mean_std(train_dataloader)
# print(f"Calculated mean: {mean}")
# print(f"Calculated std: {std}")
"""

# %%
import torchvision.transforms.v2 as transforms

MRI_image_transform_train = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(degrees=90)], p=0.5),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        #transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #),  # ImageNet normalization
        # transforms.Normalize(
        #    mean=[0.0519, 0.0519, 0.0519], std=[1.0309, 1.0309, 1.0309]
        # ),  # ImageNet normalization
        # transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # ),  # ImageNet normalization
    ]
)

MRI_image_transform_test = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # ),  # ImageNet normalization
        # transforms.Normalize(
        #    mean=[0.0519, 0.0519, 0.0519], std=[1.0309, 1.0309, 1.0309]
        # ),  # ImageNet normalization
        # transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # ),  # ImageNet normalization
    ]
)

# %%
# Visualize the transformed image
# np.random.seed(42)
indice = np.random.randint(0, len(df))

image = Image.open(df.iloc[indice]["image_path"])
tumor = df.iloc[indice]["tumor_type"]
index = indice
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis("off")
plt.title(f"Index: {index}\n{tumor}")
plt.show()

image = MRI_image_transform_train(image)
plt.figure(figsize=(6, 6))
plt.imshow(image.permute(1, 2, 0))
plt.axis("off")
plt.title(f"Index: {index}\n{tumor}")
plt.show()


# %%
############################
# Augmentation
############################

train_dataset = BrainTumorDataset(df_train, transform=MRI_image_transform_train)
test_dataset = BrainTumorDataset(df_test, transform=MRI_image_transform_test)


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
"""
############################
# Model1
############################

class BrainTumorClassifier(nn.Module):
    def __init__(self):
        super(BrainTumorClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = BrainTumorClassifier().to(device)
print(model)
"""

# %%

############################
# Model 2
############################
# This is training model B in the slide deck
class BrainTumorClassifierV2(nn.Module):
    def __init__(self):
        super(BrainTumorClassifierV2, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc_relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        # Applying first convolutional layer
        x = self.pool1(self.relu1(self.conv1(x)))
        # Applying second convolutional layer
        x = self.pool2(self.relu2(self.conv2(x)))
        # Applying third convolutional layer
        x = self.pool3(self.relu3(self.conv3(x)))

        # Flattening the last feature map for the fully connected layer
        x = x.view(-1, 128 * 16 * 16)
        x = self.fc_relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = BrainTumorClassifierV2().to(device)
# print(model)


# %%
"""
############################
# Model 3
############################
torch.cuda.empty_cache()
import torchvision

num_classes = 4

model = torchvision.models.mobilenet_v3_large(pretrained=True)

num_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features=num_features, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=num_classes, bias=True),
)
model = model.to(device)
print(next(model.parameters()).device)
"""

# %%

############################
# Model 4
############################
# This is the training model A in the slide deck
torch.cuda.empty_cache()
class ConvNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1
        )

        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features=64 * 64 * 32, out_features=num_classes)

        # Feed forwad function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = output.view(-1, 32 * 64 * 64)
        output = self.fc(output)

        return output

model = ConvNet(num_classes=4).to(device)


# %%
"""
############################
# Model 5
############################
import torch
import torch.nn as nn
from torchvision import models
torch.cuda.empty_cache()

def create_resnet50_model(num_classes=4):
    # Load a pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    # Modify the last fully connected layer to match the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


# Create the model
model = create_resnet50_model()

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfer the model to the device
model = model.to(device)

# Print the modified model
print(model)
"""


# %%
"""
############################
# Model 6
############################
import torch
import torch.nn as nn
from torchvision import models
torch.cuda.empty_cache()

def create_vgg19_model(num_classes=4):
    # Load a pre-trained VGG-19 model
    model = models.vgg19(pretrained=True)

    # Modify the classifier - change the last layer
    num_features = model.classifier[6].in_features  # The last layer's in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)

    return model


# Create the model
model = create_vgg19_model()

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfer the model to the device
model = model.to(device)

# Print the modified model
print(model)
"""

# %%

############################
# Model 7
############################
# This is model C in the slide deck
from torchvision.models import efficientnet_b0

# Create the model
model = efficientnet_b0(num_classes=4).to(device)


# %%
############################
# Optimizer and Loss Function
############################
# %%
# loss funcion
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# %%
############################
# Training
############################

# Choose the model
torch.cuda.empty_cache()
while True:
    choose_model = input("Please input the model identity from A, B, and C: ")
    if choose_model == 'A':
        model = ConvNet(num_classes=4).to(device)
    elif choose_model == 'B':
        model = BrainTumorClassifierV2().to(device)
    elif choose_model == 'C':
        model = efficientnet_b0(num_classes=4).to(device)
    else:
        print("Please input the correct model identity!")
        continue
    break
print(model)
# %%

epochs = 50

# torch.manual_seed(42)
loss_train_history = []
accuracy_train_history = []
loss_test_history = []
accuracy_test_history = []

for epoch in range(epochs):

    model.train()
    loss_train = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_dataloader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)

        loss_train += loss.item()

        loss.backward()

        optimizer.step()

    accuracy = 100 * correct / total
    print(f"Epoch: {epoch}, Train Accuracy: {accuracy}")    
    print(f"Epoch: {epoch}, Train Loss: {loss_train/len(train_dataloader)}")
    accuracy_train_history.append(accuracy)
    loss_train_history.append(loss_train / len(train_dataloader))

    if epoch % 1 == 0:
        model.eval()

        correct = 0
        total = 0
        loss_test = 0

        with torch.no_grad():

            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                loss_test += criterion(outputs, labels).item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(f"Epoch: {epoch}, Test Loss: {loss_test/len(test_dataloader)}")
        loss_test_history.append(loss_test / len(test_dataloader))
        accuracy_test_history.append(accuracy)
        print(f"Epoch: {epoch}, Test Accuracy: {accuracy}")


# %%

# Save the model
torch.save(model.state_dict(), f"model{choose_model}_Adam_lr=0.001_train1.pth")


# %%

# Load the model
model.load_state_dict(torch.load(f"model{choose_model}_Adam_lr=0.001_train1.pth"))

# %%
############################
# Visulization of the prediction
############################


# %%
"""
image = Image.open(df_test.iloc[0]["image_path"])
plt.imshow(image)
plt.axis("off")
label = df_test.iloc[0]["label"]
image = MRI_image_transform_test(image)

output = model(image.unsqueeze(0).to(device))

_, predicted_label = torch.max(output, 1)
print(label, predicted_label.item())
"""


# %%
# Visualize the prediction
def prediction_plots(df, indices):
    fig, axs = plt.subplots(1, 4, figsize=(10, 10))

    for i, index in enumerate(indices):
        image = Image.open(df.iloc[index]["image_path"])
        ax = axs[i]
        ax.imshow(image)
        ax.axis("off")
        image = MRI_image_transform_test(image)
        output = model(image.unsqueeze(0).to(device))

        tumor = df.iloc[index]["tumor_type"]
        predicted_label = label_to_tumer_type(torch.argmax(output).item())
        ax.set_title(f"Index: {index}\n{tumor}\n{predicted_label}")

    plt.tight_layout()
    plt.show()

# %%
# np.random.seed(42)
for i in range(4):
    indices = np.random.randint(0, len(df_test), 4)
    prediction_plots(df_test, indices)

# %%
# Visualize the loss history
plt.figure(figsize=(6, 4))
plt.plot(range(epochs), loss_train_history, label="Train Loss")
plt.plot(range(epochs), loss_test_history, label="Test Loss")
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper left")
plt.show()

# %%
# Visualize the accuracy history
plt.figure(figsize=(6, 4))
plt.plot(range(epochs), accuracy_train_history, label="Train Accuracy")
plt.plot(range(epochs), accuracy_test_history, label="Test Accuracy")

plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='upper left')
plt.show()


# %%
