import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import AlexNet
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
writer = SummaryWriter(log_dir='logs', flush_secs=60)
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

image_path = 'F:\Old directory'
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
train_num = len(train_dataset)


flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())

json_str = json.dumps(cla_dict, indent=1)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
print('Using {} dataloader workers every process'.format(nw))


train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)


validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=4, shuffle=True,
                                              num_workers=0)

print("using {} images for training, {} images for validation.".format(train_num,
                                                                       val_num))
#test_data_iter = iter(validate_loader)
#test_image, test_label = test_data_iter.next()


#def imshow(img):
#    img = img / 2 + 0.5  # unnormalize
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()


#print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
#imshow(utils.make_grid(test_image))

net = AlexNet(num_classes=2, init_weights=True)

net.to(device)
loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.0002)

epochs = 10
save_path = './AlexNet2.pth'
best_acc = 0.0
train_steps = len(train_loader)
validate_steps = len(validate_loader)
for epoch in range(epochs):

    net.train()
    running_loss = 0.0
    train_acc = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_acc += torch.eq(torch.max(net(images.to(device)), dim=1)[1], labels.to(device)).sum().item()

        #train_y = torch.max(outputs, dim=1)[1]
        #train_acc += torch.eq(train_y, labels.to(device)).sum().item()
        #train_acc2 = train_acc / train_num




        train_bar.desc = "train epoch[{}/{}] train loss: {:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    writer.add_scalar('Train_loss', running_loss / train_steps, epoch)
    writer.add_scalar('Train_acc', train_acc / len(train_dataset), epoch)


    net.eval()
    acc = 0.0
    val_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            #loss = loss_function(outputs, val_labels.to(device))
            val_loss += loss.item()
            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       epochs)
            #vali_loss = val_loss / validate_steps

    val_accurate = acc / val_num
    vali_loss = val_loss / len(validate_loader)
    #print('[epoch %d] val_loss: %.3f  val_accuracy: %.3f' %
    #      (epoch + 1, vali_loss, val_accurate))
    print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_loss: %.3f  val_accuracy: %.3f' % (
        epoch + 1, running_loss / train_steps, train_acc / len(train_dataset), vali_loss, val_accurate))

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

print('Finished Training')