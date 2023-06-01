# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from CustomDataset import CustomDataset
# from CNNLSTMModel import CNNLSTMModel
# from torchvision import transforms
# from tqdm import tqdm
#
# transform = transforms.Compose([
#             # transforms.Resize((720, 1280)),
#             transforms.Resize((190, 320)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#
# def calculate_class_accuracy(outputs, labels):
#     _, predicted = torch.max(outputs.data, 1)
#     total = labels.size(0)
#     correct = (predicted == labels).sum().item()
#
#     class_accuracy = torch.zeros(labels.max() + 1)
#     for c in range(labels.max() + 1):
#         class_total = (labels == c).sum().item()
#         class_correct = ((predicted == labels) & (labels == c)).sum().item()
#         class_accuracy[c] = (class_correct / class_total) * 100 if class_total != 0 else 0
#
#     return correct, total, class_accuracy
# def main():
#     # 메타데이터 파일과 프레임 폴더 경로를 설정합니다.
#     metadata_file = 'metadata_train.json'
#     metaval_file = 'metadata_val.json'
#     frames_folder = '/home/dnc/Desktop/mnt/workspace/datasets/DoTA_dataset/frames'
#
#     # 커스텀 데이터셋의 인스턴스를 생성합니다.
#     train_dataset = CustomDataset(metadata_file, frames_folder, transform)
#     val_dataset = CustomDataset(metaval_file, frames_folder, transform)
#
#     # 데이터셋을 훈련 및 검증 세트로 나눕니다.
#     # train_ratio = 0.8
#     # train_size = int(train_ratio * len(dataset))
#     # val_size = len(dataset) - train_size
#     # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#
#     # 훈련 데이터로 데이터 로더를 생성합니다.
#     # batch_size = 4
#     batch_size = 64
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
#     # 검증 데이터로 데이터 로더를 생성합니다.
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
#     # 모델의 파라미터를 정의합니다.
#     hidden_size = 256
#     num_layers = 2
#
#     # CNNLSTMModel의 인스턴스를 생성합니다.
#     model = CNNLSTMModel(hidden_size, num_layers)
#     print(model)
#
#     # 모델을 실행할 장치(device)를 설정합니다.
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device_ids = [0, 1]  # 사용할 GPU 장치 ID 리스트
#
#     # 모델을 장치로 이동시킵니다.
#     model = nn.DataParallel(model, device_ids=device_ids).to(device)
#
#     # 손실 함수와 옵티마이저를 정의합니다.
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#
#     # 훈련 및 검증 과정을 진행합니다.
#     num_epochs = 30
#     for epoch in range(num_epochs):
#         # 훈련 과정
#         model.train()
#         train_loss = 0.0
#         correct = 0
#         total = 0
#         pbar = tqdm(train_dataloader, total=len(train_dataloader))
#         for batch_images, batch_labels in pbar:
#             batch_images = batch_images.to(device)
#             batch_labels = batch_labels.to(device)
#
#             # 순전파(forward pass)
#             outputs = model(batch_images)
#
#             # 손실 계산
#             loss = criterion(outputs, batch_labels)
#
#             # 역전파(backward pass) 및 최적화(optimizer)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item() * batch_images.size(0)
#
#             # 정확도 계산
#             _, predicted = torch.max(outputs.data, 1)
#             total += batch_labels.size(0)
#             correct += (predicted == batch_labels).sum().item()
#
#             # 진행 상황 업데이트
#             pbar.set_postfix({'Train Loss': loss.item(), 'Train Acc': (correct / total) * 100})
#
#         # 훈련 손실과 정확도를 계산합니다.
#         train_loss = train_loss / len(train_dataset)
#         train_acc = correct / total
#
#         # 검증 과정
#         model.eval()
#         val_loss = 0.0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             pbar = tqdm(val_dataloader, total=len(val_dataloader))
#             for batch_images, batch_labels in pbar:
#                 batch_images = batch_images.to(device)
#                 batch_labels = batch_labels.to(device)
#
#                 # 순전파(forward pass)
#                 outputs = model(batch_images)
#
#                 # 손실 계산
#                 loss = criterion(outputs, batch_labels)
#                 val_loss += loss.item() * batch_images.size(0)
#
#                 # 정확도 계산
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += batch_labels.size(0)
#                 correct += (predicted == batch_labels).sum().item()
#
#                 # 진행 상황 업데이트
#                 pbar.set_postfix({'Val Loss': loss.item(), 'Val Acc': (correct / total) * 100})
#
#         # 검증 손실과 정확도를 계산합니다.
#         val_loss = val_loss / len(val_dataset)
#         val_acc = correct / total
#
#         # 출력: 훈련 손실, 검증 손실, 훈련 정확도, 검증 정확도
#         print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
#               f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
#
#         torch.save(model.state_dict(), f'output/model_epoch{epoch + 1}.pt')
#
# if __name__ == '__main__':
#     main()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
from CNNLSTMModel import CNNLSTMModel
from torchvision import transforms
from tqdm import tqdm


transform = transforms.Compose([
    # transforms.Resize((720, 1280)),
    transforms.Resize((190, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def calculate_class_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()

    class_accuracy = torch.zeros(labels.max() + 1)
    for c in range(labels.max() + 1):
        class_total = (labels == c).sum().item()
        class_correct = ((predicted == labels) & (labels == c)).sum().item()
        class_accuracy[c] = (class_correct / class_total) * 100 if class_total != 0 else 0

    return correct, total, class_accuracy


def main():
    # 메타데이터 파일과 프레임 폴더 경로를 설정합니다.
    metadata_file = '../metadata_train.json'
    metaval_file = '../metadata_val.json'
    frames_folder = '/home/dnc/Desktop/mnt/workspace/datasets/DoTA_dataset/frames'

    # 커스텀 데이터셋의 인스턴스를 생성합니다.
    train_dataset = CustomDataset(metadata_file, frames_folder, transform)
    val_dataset = CustomDataset(metaval_file, frames_folder, transform)

    # 데이터셋을 훈련 및 검증 세트로 나눕니다.
    # train_ratio = 0.8
    # train_size = int(train_ratio * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 훈련 데이터로 데이터 로더를 생성합니다.
    # batch_size = 4
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 검증 데이터로 데이터 로더를 생성합니다.
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델의 파라미터를 정의합니다.
    hidden_size = 256
    num_layers = 2

    # CNNLSTMModel의 인스턴스를 생성합니다.
    model = CNNLSTMModel(hidden_size, num_layers)
    print(model)

    # 모델을 실행할 장치(device)를 설정합니다.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_ids = [0, 1]  # 사용할 GPU 장치 ID 리스트

    # 모델을 장치로 이동시킵니다.
    model = nn.DataParallel(model, device_ids=device_ids).to(device)

    # 손실 함수와 옵티마이저를 정의합니다.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 훈련 및 검증 과정을 진행합니다.
    num_epochs = 15
    for epoch in range(num_epochs):
        # 훈련 과정
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch_images, batch_labels in pbar:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # 순전파(forward pass)
            outputs = model(batch_images)

            # 손실 계산
            loss = criterion(outputs, batch_labels)

            # 역전파(backward pass) 및 최적화(optimizer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_images.size(0)

            # 정확도 계산
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

            # 진행 상황 업데이트
            pbar.set_postfix({'Train Loss': loss.item(), 'Train Acc': (correct / total) * 100})

        # 훈련 손실과 정확도를 계산합니다.
        train_loss = train_loss / len(train_dataset)
        train_acc = correct / total

        # 검증 과정
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(val_dataloader, total=len(val_dataloader))
            for batch_images, batch_labels in pbar:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                # 순전파(forward pass)
                outputs = model(batch_images)

                # 손실 계산
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_images.size(0)

                # 정확도 계산
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

                # 진행 상황 업데이트
                pbar.set_postfix({'Val Loss': loss.item(), 'Val Acc': (correct / total) * 100})

        # 검증 손실과 정확도를 계산합니다.
        val_loss = val_loss / len(val_dataset)
        val_acc = correct / total

        # 클래스별 정확도 계산
        _, _, class_accuracy = calculate_class_accuracy(outputs, batch_labels)

        # 출력: 훈련 손실, 검증 손실, 훈련 정확도, 검증 정확도, 클래스별 정확도
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        print('Class Accuracy:', class_accuracy)

        torch.save(model.state_dict(), f'output/model_epoch{epoch + 1}.pt')


if __name__ == '__main__':
    main()
