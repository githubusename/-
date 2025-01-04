import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from model import CustomCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# 定义数据集类
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 遍历 NORMAL 和 PNEUMONIA 文件夹
        for label, category in enumerate(['NORMAL', 'PNEUMONIA']):
            category_path = os.path.join(folder_path, category)
            for file_name in os.listdir(category_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(category_path, file_name))
                    self.labels.append(label)  # NORMAL -> 0, PNEUMONIA -> 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')  # 确保是RGB格式

        if self.transform:
            img = self.transform(img)

        return img, label


# 定义与训练时相同的预处理
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),  # 如果您的模型是灰度图像输入
    transforms.ToTensor(),
])

# 加载验证集
val_folder_path = 'input/chest-xray-pneumonia/val'  # 替换为您的验证集文件夹路径
val_dataset = ImageDataset(val_folder_path, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN().to(device)
model.load_state_dict(torch.load('custom_cnn_model.pth', weights_only=True))  # 加载模型权重
model.eval()  # 设置为评估模式


# 定义验证函数
def validate_model(model, val_loader):
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # 收集标签和预测结果
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算各种评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")


# 调用验证函数
validate_model(model, val_loader)
