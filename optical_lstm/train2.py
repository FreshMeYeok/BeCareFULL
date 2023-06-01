import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CustomDataset_cellNum import CustomDataset
from OpticalLSTM_cellNum import OpticalLSTM
from torchvision import transforms
from tqdm import tqdm
import os

transform = transforms.Compose([
    transforms.Resize((180, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.229]),
])

def main():
    metadata_file = '../metadata_train.json'
    metaval_file = '../metadata_val.json'
    frames_folder = '/home/dnc/Desktop/mnt/workspace/datasets/DoTA_dataset/frames'

    train_dataset = CustomDataset(metadata_file, frames_folder, transform)
    val_dataset = CustomDataset(metaval_file, frames_folder, transform)

    batch_size = 8
    print("Create Dataset Start")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Create Dataset End")

    hidden_size = 128
    num_classes = 2

    model = OpticalLSTM(180*320, hidden_size, num_classes)
    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_ids = [0, 1]

    model = nn.DataParallel(model, device_ids=device_ids).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch_images, batch_labels in pbar:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)

            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

            # 각 클래스의 정확도 업데이트
            for label, prediction in zip(batch_labels, predicted):
                class_correct[label] += (prediction == label).item()
                class_total[label] += 1

            train_acc = (correct / total) * 100

            # 진행 상황 업데이트
            pbar.set_postfix({'Train Loss': loss.item(), 'Train Acc': train_acc})
            pbar.set_postfix_str(get_class_accuracy(class_correct, class_total))

        train_loss = train_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            pbar = tqdm(val_dataloader, total=len(val_dataloader))
            for batch_images, batch_labels in pbar:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_images)

                loss = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

                for label, prediction in zip(batch_labels, predicted):
                    class_correct[label] += (prediction == label).item()
                    class_total[label] += 1

                val_acc = (correct / total) * 100

                pbar.set_postfix({'Val Loss': loss.item(), 'Val Acc': val_acc})
                pbar.set_postfix_str(get_class_accuracy(class_correct, class_total))

        val_loss = val_loss / len(val_dataset)
        val_acc = correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc >= best_acc:
            best_acc = val_acc
            filename = f'output/train2/model_epoch{epoch + 1}.pt'
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            torch.save(model.state_dict(), filename)

def get_class_accuracy(class_correct, class_total):
    class_acc_str = ', '.join([f'Class {i} Acc: {correct / total * 100:.2f}%'
                               for i, (correct, total) in enumerate(zip(class_correct, class_total))])
    return class_acc_str

if __name__ == '__main__':
    main()
