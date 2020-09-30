import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# parameters
user = 'dango'

# data
l = torch.randn(128, 6, 3)
v = torch.randn(128, 7, 4)
a = torch.randn(128, 8, 5)
gt = []
for i in range(128):
    gt.append(random.randint(0, 5))
gt = torch.as_tensor(gt)
# print(l)
# print(v)
# print(a)
# print(gt)

def data_loader(linguistic, visual, acoustic, ground_truth, batch_size=2):
    count = 0
    while count < len(linguistic):
        batch = []
        if batch_size < len(linguistic) - count:
            size = batch_size
        else: 
            size = len(linguistic) - count
        batch.append((linguistic[count: count+size], visual[count: count+size], acoustic[count: count+size], ground_truth[count: count+size]))
        count += size
        yield batch

dl = data_loader(l, v, a, gt)
# print(next(dl))

# model
class fenlei_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 16)
        self.norm1 = nn.LayerNorm(16, eps=1e-12)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(16, 6)
    def forward(self, linguistic, visual, acoustic):
        linguistic = torch.mean(linguistic, 1)
        visual = torch.mean(visual, 1)
        acoustic = torch.mean(acoustic, 1)
        fusion = torch.cat((linguistic, visual, acoustic), 1)
        fusion = self.activ1(self.norm1(self.fc1(fusion)))
        fusion = self.classifier(fusion)
        return fusion

# print(fenlei_model()(l,v,a).shape)

# run
def train(model, iterator, optimizer):
    model.train()
    epoch_loss, count = 0, 0
#     iter_bar = tqdm(iterator, desc='Training')
#     for _, batch in enumerate(iter_bar):
    for _, batch in enumerate(iterator):
        count += 1
        optimizer.zero_grad()
        linguistic, visual, acoustic, ground_truth = zip(*batch)
        logits_clsf = model(linguistic[0].cuda(), visual[0].cuda(), acoustic[0].cuda())
        loss = nn.CrossEntropyLoss()(logits_clsf, ground_truth[0].long().cuda())
        loss = loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #梯度裁剪
        optimizer.step()
#         iter_bar.set_description('Iter (loss=%4.4f)'%loss.item())
        epoch_loss += loss.item()
    return epoch_loss / count

def valid(model, iterator):
    model.eval()
    epoch_loss, count = 0, 0
    with torch.no_grad():
#         iter_bar = tqdm(iterator, desc='Validation')
#         for _, batch in enumerate(iter_bar):
        for _, batch in enumerate(iterator):
            count += 1
            linguistic, visual, acoustic, ground_truth = zip(*batch)
            logits_clsf = model(linguistic[0].cuda(), visual[0].cuda(), acoustic[0].cuda())
            loss = nn.CrossEntropyLoss()(logits_clsf, ground_truth[0].long().cuda())
            loss = loss.mean()
#             iter_bar.set_description('Iter (loss=%4.4f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss / count

def train_model(model, linguistic, visual, acoustic, ground_truth, batch_size=2, learning_rate=0.01, epochs=200):
    writer = SummaryWriter('/home/dango/multimodal/log/')
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(epochs):
        train_iterator = data_loader(linguistic, visual, acoustic, ground_truth, batch_size)
        valid_iterator = data_loader(linguistic, visual, acoustic, ground_truth, batch_size)
#         print('Epoch: ' + str(epoch+1))
        train_loss = train(model, train_iterator, optimizer)
        valid_loss = valid(model, valid_iterator)
        writer.add_scalar('Loss_value', valid_loss, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss) 
        if valid_loss == min(loss_list):
            stop = 0
            print(epoch+1, valid_loss)
        else:
            stop += 1
            if stop > 5:
                break
    return min(loss_list)

cls_loss = train_model(fenlei_model().to(device), l, v, a, gt)
