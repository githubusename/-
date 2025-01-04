import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix

def create_dataset(folder_path):
    my_list = []
    for category in ['NORMAL', 'PNEUMONIA']: #遍历这两个类
        category_path = os.path.join(folder_path, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            # 确保我们只添加图像文件
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                my_list.append([file_path, category])
    return pd.DataFrame(my_list, columns=['file_path', 'label'])

# 数据路径
dataset_dir = 'input/chest-xray-pneumonia/chest_xray'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')

#为训练、验证和测试数据集创建dataframe
train_df = create_dataset(train_dir)
val_df = create_dataset(val_dir)
test_df = create_dataset(test_dir)

#将标签转换为数字：NORMAL -> 0, PNEUMONIA -> 1
train_df['label'] = train_df['label'].map({'NORMAL': 0, 'PNEUMONIA': 1})
val_df['label'] = val_df['label'].map({'NORMAL': 0, 'PNEUMONIA': 1})
test_df['label'] = test_df['label'].map({'NORMAL': 0, 'PNEUMONIA': 1})

#打印数据集大小
print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}, Test set size: {len(test_df)}")

#对给定DataFrame中的类别进行计数
def count_categories(df, dataset_name):
    category_counts = df['label'].value_counts()
    print(f"{dataset_name} set:")
    print(f"  NORMAL: {category_counts.get(0, 0)}")
    print(f"  PNEUMONIA: {category_counts.get(1, 0)}")

#计数和显示训练，验证和测试数据集
print("Image Counts per Category:")
count_categories(train_df, "Train")
count_categories(val_df, "Validation")
count_categories(test_df, "Test")


#可视化类分布
train_counts = train_df['label'].value_counts()
val_counts = val_df['label'].value_counts()
test_counts = test_df['label'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(['Normal', 'Pneumonia'], train_counts)
plt.title("Training Data Class Distribution")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(['Normal', 'Pneumonia'], val_counts)
plt.title("Validation Data Class Distribution")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(['Normal', 'Pneumonia'], test_counts)
plt.title("Test Data Class Distribution")
plt.ylabel("Count")
plt.show()

#检查多gpu可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

#为多gpu使用准备模型的函数
def prepare_model_for_multigpu(model):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    return model

#定义Dataset类
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

#数据预处理
train_transform_cnn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform_cnn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])


#创建数据集
train_dataset_cnn = ImageDataset(train_df, transform=train_transform_cnn)
val_dataset_cnn = ImageDataset(val_df, transform=val_transform_cnn)



# DataLoader
batch_size = 8
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=True)
val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=batch_size, shuffle=False)


#自定义CNN模型
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel for grayscale
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_from_scratch = CustomCNN().to(device)
model_from_scratch = prepare_model_for_multigpu(model_from_scratch)


#训练功能与历史跟踪
from tqdm import tqdm

from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import torch

def train_model_with_metrics(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_recalls = []
    val_recalls = []
    train_f1_scores = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        # 训练循环与tqdm进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", ncols=100, leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集标签和预测值
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # 更新进度条损失和准确性
            train_bar.set_postfix(loss=running_loss / (train_bar.n + 1), accuracy=100 * correct / total)

        # 计算训练损失和准确性
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 计算训练召回率和F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
        train_recalls.append(recall * 100)
        train_f1_scores.append(f1 * 100)

        # 验证循环与tqdm进度条
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_predictions = []

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", ncols=100, leave=False)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # 收集标签和预测值
                val_labels.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

                val_bar.set_postfix(loss=val_loss / (val_bar.n + 1), accuracy=100 * val_correct / val_total)

        # 计算验证损失和准确性
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 计算验证召回率和F1分数
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_predictions, average='macro')
        val_recalls.append(val_recall * 100)
        val_f1_scores.append(val_f1 * 100)

        # 打印epoch的结果
        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train Recall: {recall * 100:.2f}%, Train F1: {f1 * 100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val Recall: {val_recall * 100:.2f}%, Val F1: {val_f1 * 100:.2f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies, train_recalls, val_recalls, train_f1_scores, val_f1_scores


#为自定义CNN定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer_from_scratch = optim.Adam(model_from_scratch.parameters(), lr=0.001)

#训练自定义CNN模型
print("Training Custom CNN Model ...")
train_losses_cnn, val_losses_cnn, train_accuracies_cnn, val_accuracies_cnn, \
train_recalls_cnn, val_recalls_cnn, train_f1_scores_cnn, val_f1_scores_cnn = train_model_with_metrics(
    model_from_scratch, train_loader_cnn, val_loader_cnn, criterion, optimizer_from_scratch, num_epochs=10
)

# 保存模型
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

# 保存训练好的模型
model_filename = 'custom_cnn_model.pth'
save_model(model_from_scratch, model_filename)

#绘制自定义CNN的训练和验证损失/精确度
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, model_name):
    epochs = range(1, len(train_losses) + 1)

    # 绘制损失函数
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制精确度
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

#绘制自定义CNN的结果
plot_training_history(train_losses_cnn, val_losses_cnn, train_accuracies_cnn, val_accuracies_cnn, "Custom CNN ")