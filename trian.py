from model import resnet34
import torch
from torchvision import transforms, datasets
import torch.utils.data.dataloader as Data
import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as D
import os
import random
from tqdm import tqdm
from PIL import Image
import json
import torch.nn as nn
import sys


def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

cla_dict = {'AK': '光化性角化病', 'BCC': '基底细胞癌', 'BKL': '良性角化病', 'DF': '皮肤纤维瘤', 'MEL': '脱色性皮肤病',
            'NV': '黑色素细胞痣', 'SCC': '鳞状细胞癌', 'VASC': '血管病变'}
json_str = json.dumps(cla_dict, indent=8, ensure_ascii=False)
print(json_str)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

data = datasets.ImageFolder(root=os.path.join('../DATA/SKIN'),
                            transform=transform)
print(len(data))

train_data, val_data, test_data = [], [], []
test_txt = []
labels = {'AK': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'SCC': 6, 'VASC': 7}
skin_str = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
for label in labels:
    print(label)
    files_path = '../DATA/SKIN/' + label
    files_list = os.listdir(files_path)
    print(len(files_list))
    # train_file, test_file = data_split(files_list, 0.8)
    # test_file, val_file = data_split(test_val_file, 0.5)
    # print(len(train_file), len(test_file), len(val_file))
    train_file, val_file = data_split(files_list, 0.8)
    print(len(train_file), len(val_file))
    for file in train_file:
        img = Image.open('../DATA/SKIN/' + label + '/' + file)
        img = transform(img)
        train_data.append(tuple([img, torch.tensor(labels[label])]))
    for file in val_file:
        img = Image.open('../DATA/SKIN/' + label + '/' + file)
        img = transform(img)
        val_data.append(tuple([img, torch.tensor(labels[label])]))
    # for file in test_file:
    #     img = Image.open('../DATA/SKIN/' + label + '/' + file)
    #     test_txt.append([file, labels[label]])
    #     img = transform(img)
    #     test_data.append(tuple([img, torch.tensor(labels[label])]))

with open('data3.txt', 'w') as f:
    for i in test_txt:
        f.write(str(i[0]) + ' ' + str(i[1]))
        f.write('\r\n')
f.close()

print("train, val", len(train_data), len(val_data))
train_loader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, drop_last=True)
val_loader = Data.DataLoader(dataset=val_data, batch_size=16, shuffle=False, drop_last=True)

net = resnet34()
net.to(device)

model_weight_path = './resnet34-pre.pth'
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location=device))
for param in net.parameters():
    param.requires_grad = False

in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 8)

loss_function = nn.CrossEntropyLoss()
params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=1e-3)

epochs = 10
best_acc = 0.0
save_path = 'ResNet34_0.613.pth'
train_steps = len(train_loader)
val_steps = len(val_loader)
summaryWriter = SummaryWriter(log_dir="./log/log_34_epoch_10")
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

    net.eval()
    acc = 0.0
    running_val_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for data in val_bar:
            val_images, val_labels = data
            outputs = net(val_images.to(device))
            val_loss = loss_function(outputs, val_labels.to(device))
            running_val_loss += val_loss.item()
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       epochs)
    val_accurate = acc / len(val_data)
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, val_accurate))

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

    summaryWriter.add_scalars("epoch_loss", {"train_loss": running_loss / train_steps,
                                             "val_loss": running_val_loss / val_steps}, epoch + 1)
    summaryWriter.add_scalars("epoch_acc", {"acc": val_accurate}, epoch + 1)

print("val best acc", best_acc)
print('Finished Training!')

# test_loader = Data.DataLoader(dataset=test_data, batch_size=16, shuffle=False, drop_last=True)
# with torch.no_grad():
#     acc_num = 0.0
#     for data in test_loader:
#         images, labels = data
#         pre = net(images.to(device))
#         predict_y = torch.max(pre, dim=1)[1]
#         acc_num += torch.eq(predict_y, labels.to(device)).sum().item()
#     accurate = acc_num / len(test_data)
#     print("test acc:", accurate)
