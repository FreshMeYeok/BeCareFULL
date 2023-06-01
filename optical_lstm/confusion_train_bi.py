import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset4test_bi import CustomDataset
from OpticalLSTM_BI import OpticalLSTM
from torchvision import transforms
from tqdm import tqdm
import os
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((180, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.229]),
])
def main():
    metadata_file = '../metadata_val.json'
    frames_folder = '/home/dnc/Desktop/mnt/workspace/datasets/DoTA_dataset/frames'

    # 커스텀 데이터셋의 인스턴스를 생성합니다.
    val_dataset = CustomDataset(metadata_file, frames_folder, transform)

    # 데이터셋을 훈련 및 검증 세트로 나눕니다.
    # train_ratio = 0.8
    # train_size = int(train_ratio * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 훈련 데이터로 데이터 로더를 생성합니다.
    # batch_size = 4
    batch_size = 8
    # train_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # 검증 데이터로 데이터 로더를 생성합니다.
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the path to the pre-trained model file
    model_path = '/home/dnc/Desktop/test/optical_lstm/output/train_bi/model_epoch6.pt'
    hidden_size = 128
    num_layers = 2

    # Create an instance of the CNNLSTMModel
    model = OpticalLSTM(180*320, hidden_size, num_layers)

    # Load the pre-trained model state_dict
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    # Modify the pre-trained model's state_dict to match the model's definition
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)

    # Load the modified state_dict into the model
    model.load_state_dict(model_dict)

    # 검증 단계
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_predictions = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in tqdm(val_dataloader):
            # Forward pass
            # images = images.to(device)
            # labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            val_loss += loss.item()
            batch_accuracy = (predicted == labels).sum().item() / labels.size(0)
            # print(f'Batch Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}')

    auc = roc_auc_score(all_labels, all_predictions)
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    # 평균 검증 손실과 정확도 계산
    val_loss = val_loss / len(val_dataloader)
    val_accuracy = val_correct / val_total

    # 결과 출력
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}', f"AUC: {auc:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    main()
