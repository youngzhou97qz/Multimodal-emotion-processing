import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# parameters
# Love,Anxiety,Sorrow,Joy,Expect,Hate,Anger,Surprise,Polarity
user_dir = '/home/dango/multimodal/ren/'
log_dir = user_dir + 'log/cg_01/'
EPOCHS = 999
BATCH = 64
DIM = 780
LR = 0.01

# data
def label_list(num):
    label_list, count_list = [], []
    with open(user_dir + '1487_txt_hier_sents_202002/cet_' + str(num) + '.txt', 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            if line[0] == 's':
                count += 1
                if line.split(':')[2] == '\n' or line.split(':')[2] == '/n\n' or line.split(':')[2] == '/n' or line.split(':')[2] == '' or line.split(':')[2][0] == '/':
                    count_list.append(count)
                    continue
                else:
                    label = [0,0,0,0,0,0,0,0,0]
                    line = line.split(':')[1]
                    line = line.split(',')[:8]
                    for index, x in enumerate(line):
                        if x != '0.0':
                            label[index] = 1
                    if sum(label) == 0:
                        label == [0,0,0,0,0,0,0,0,1]
                    label_list.append(label)
    return label_list, count_list

def para_sent_list(num, count_list):
    para_list, sent_list = [], []
    with open(user_dir + '1487_xml_doc_segmented_utf8/cet_' + str(num) + '.xml', 'r') as f:
        lines = f.readlines()
        count = 0
        for index, line in enumerate(lines):
            if '<S_no>' in line:
                count += 1
                if count in count_list:
                    continue
                else:
                    para_list.append(line.split('段第')[0].split('第')[1])
                    sent_list.append(line.split('段第')[1].split('句')[0])
    return para_list, sent_list

def data_set(catagory='test'):
    if catagory == 'test':
        start, end = 1189, 1488
    else:
        start, end = 1, 1189
    data_set = []
    for i in tqdm(range(start, end)):
        if i == 490 or i == 761:
            continue
        labe_list, count_list = label_list(i)
        para_list, sent_list = para_sent_list(i, count_list)
        array = np.load(user_dir + '1487_npy_12emos_768bert_flatten_sents_202002/cet_' + str(i) + '.txt_12emos_768Bert.npy')
        for j in range(len(para_list)):
            data_set.append({'name':str(i)+'_'+para_list[j]+'_'+sent_list[j], 'feat':array[j], 'label':labe_list[j]})
    return data_set

train_set = data_set('train')
test_set = data_set('test')

def data_loader(name_list, batch_size):
    random.shuffle(name_list)
    count = 0
    while count < len(name_list):
        batch = []
        size = min(batch_size, len(name_list) - count)
        for _ in range(size):
            feat_temp, mask_temp = [], []
            for temp in name_list:
                if temp['name'].split('_')[0] == name_list[count]['name'].split('_')[0] and temp['name'].split('_')[1] == name_list[count]['name'].split('_')[1]\
                and int(temp['name'].split('_')[2]) == (int(name_list[count]['name'].split('_')[2])-1):
                    feat = temp['feat']
                    feat_temp.append(np.expand_dims(feat, axis=0))
                    mask_temp.append(1)
                    feat = name_list[count]['feat']
                    feat_temp.append(np.expand_dims(feat, axis=0))
                    mask_temp.append(1)
            if len(feat_temp) == 0:
                feat = np.zeros((780))
                feat_temp.append(np.expand_dims(feat, axis=0))
                mask_temp.append(0)
                feat = name_list[count]['feat']
                feat_temp.append(np.expand_dims(feat, axis=0))
                mask_temp.append(1)
            label = name_list[count]['label']
            batch.append((np.concatenate(feat_temp, axis=0), label, np.asarray(mask_temp)))
            count += 1
        yield batch

# model
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class Control_Group(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.t1 = nn.Linear(dim, 9)
    def forward(self, feat, mask):  # (batch, 2, x_dim)
        t1_feats = feat[:,1,:]
        return self.t1(t1_feats)  # (batch, 9)

# class State_Transfer(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.t0 = nn.Linear(dim, 9)
#         self.t1 = nn.Linear(dim, 9)
#         self.alpha = nn.Parameter(torch.tensor([0.]),requires_grad=True)
#         self.trans = nn.Parameter(torch.rand(9,9), requires_grad=True)
#     def forward(self, feat, mask):  # (batch, 2, x_dim)
#         t0_feats, t1_feats = feat[:,0,:], feat[:,1,:]
#         t0_feats = self.t0(t0_feats)  # (batch, 9)
#         t1_feats = self.t1(t1_feats)  # (batch, 9)
#         out_list = []
#         for i in range(mask.shape[0]):
#             if mask[i][0] == 0:
#                 out_list.append(t1_feats[i].unsqueeze(0))
#             else:
#                 t1_out_temp = torch.matmul(t1_feats[i], torch.tanh(self.trans))
#                 t0_out_temp = (1-torch.sigmoid(self.alpha)) * t0_feats[i] + torch.sigmoid(self.alpha) * t1_out_temp
#                 out_list.append(t0_out_temp.unsqueeze(0))
#         out = torch.cat(out_list, dim=0)
#         return out  # (batch, 9)

# run
def multi_circle_loss(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1], dtype = torch.float)
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss

def train(model, iterator, optimizer):
    model.train()
    epoch_loss, count = 0, 0
    iter_bar = tqdm(iterator, desc='Training')
    for _, batch in enumerate(iter_bar):
        count += 1
        optimizer.zero_grad()
        feat, label, mask = zip(*batch)
        feat, label, mask = torch.cuda.FloatTensor(feat), torch.cuda.LongTensor(label), torch.cuda.LongTensor(mask)
        logits_clsf = model(feat, mask)
        loss = multi_circle_loss(logits_clsf, label)
        loss = loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #梯度裁剪
        optimizer.step()
        iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
        epoch_loss += loss.item()
    return epoch_loss / count

def valid(model, iterator):
    model.eval()
    epoch_loss, count = 0, 0
    with torch.no_grad():
        iter_bar = tqdm(iterator, desc='Validation')
        for _, batch in enumerate(iter_bar):
            count += 1
            feat, label, mask = zip(*batch)
            feat, label, mask = torch.cuda.FloatTensor(feat), torch.cuda.LongTensor(label), torch.cuda.LongTensor(mask)
            logits_clsf = model(feat, mask)
            loss = multi_circle_loss(logits_clsf, label)
            loss = loss.mean()
            iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss, count, epoch_loss / count

def run(model, train_list, valid_list, batch_size, learning_rate, epochs, name):
    log_file = log_dir+name+'.txt'
    with open(log_file, 'w') as log_f:
        log_f.write('epoch, train_loss, valid_loss\n')
    writer = SummaryWriter(log_dir)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(epochs):
        print('Epoch: ' + str(epoch+1))
        train_iterator = data_loader(train_list, batch_size)
        valid_iterator = data_loader(valid_list, batch_size)
        train_loss = train(model, train_iterator, optimizer)
        _, _, valid_loss = valid(model, valid_iterator)
        writer.add_scalars(name, {'train_loss':train_loss, 'valid_loss':valid_loss}, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss) 
        with open(log_file, 'a') as log_f:
            log_f.write('\n{epoch},{train_loss: 2.2f},{valid_loss: 2.2f}\n'.format(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss))
        if valid_loss == min(loss_list):
            stop = 0
            torch.save(model.state_dict(), os.path.join(log_dir, name+'_'+str(valid_loss)[:4]+'.pt'))
            if valid_loss < 0.01:
                break
        else:
            stop += 1
            if stop >= 4:
                break
    writer.close()

model_1 = Control_Group(dim=DIM).to(device)
valid_list = train_set[:5394]
train_list = train_set[5394:]
run(model_1, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_1')
model_2 = Control_Group(dim=DIM).to(device)
valid_list = train_set[5394:10795]
train_list = train_set[:5394] + train_set[10795:]
run(model_2, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_2')
model_3 = Control_Group(dim=DIM).to(device)
valid_list = train_set[10795:16188]
train_list = train_set[:10795] + train_set[16188:]
run(model_3, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_3')
model_4 = Control_Group(dim=DIM).to(device)
valid_list = train_set[16188:21596]
train_list = train_set[:16188] + train_set[21596:]
run(model_4, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_4')
model_5 = Control_Group(dim=DIM).to(device)
valid_list = train_set[21596:]
train_list = train_set[:21596]
run(model_5, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_5')
