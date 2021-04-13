import os
import math
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# parameters
# Love,Anxiety,Sorrow,Joy,Expect,Hate,Anger,Surprise,Neutral
user_dir = '/home/XX/multimodal/ren/'
log_dir = user_dir + 'log/XX/'
EPOCHS = 99
BATCH = 64
DIM = 768*3
CLIP = 1.0
DROP = 0.1
LR = 0.001

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
                    if sum(label) == 0.0:
                        label = [0,0,0,0,0,0,0,0,1]
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
        start, end = 1190, 1488
    else:
        start, end = 1, 1190
    data_set = []
    for i in range(start, end):
        if i == 490 or i == 761:
            continue
        labe_list, count_list = label_list(i)
        para_list, sent_list = para_sent_list(i, count_list)
        for j in range(len(para_list)):
            data_set.append({'name':str(i)+'_'+para_list[j]+'_'+sent_list[j], 'label':labe_list[j]})
    return data_set

train_set = data_set('train')
test_set = data_set('test')

def data_list(data_set):
    data_list, temp_list = [], []
    len_count = 0
    for dicts in data_set:
        if dicts['name'].split('_')[1] == '1' and dicts['name'].split('_')[2] == '1':
            temp_list = []
            temp_list.append({'name':'no_name'})
            temp_list.append(dicts)
        else:
            temp_list = temp_list[-1:]
            temp_list.append(dicts)
        data_list.append(temp_list)
    return data_list

train_name_list = data_list(train_set)
test_name_list = data_list(test_set)

def flatten_array(name):  # 字符级降维
    temp = np.load(user_dir + 'ren_text_feat/' + name + '.npy')
    temp_cls = temp[0]
    temp_max = np.max(temp[1:], axis=0)
    temp_avg = np.mean(temp[1:], axis=0)
    temp_feat = np.concatenate((temp_cls, temp_max, temp_avg), axis=0)
    return np.expand_dims(temp_feat, axis=0)

def data_loader(name_list, batch_size):
    random.shuffle(name_list)
    count = 0
    while count < len(name_list):
        batch = []
        size = min(batch_size, len(name_list) - count)
        for _ in range(size):
            if name_list[count][0]['name'] == 'no_name':
                feat_0 = np.zeros((1,DIM))
            else:
                feat_0 = flatten_array(name_list[count][0]['name'])
            feat_1 = flatten_array(name_list[count][1]['name'])
            feat = np.concatenate((feat_0, feat_1), axis=0)
            label = name_list[count][1]['label']
            batch.append((feat, label))
            count += 1
        yield batch

# # 转移模型
class Concat_Linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.intensity = nn.Linear(dim, 9, bias=False)
        self.stimulation = nn.Linear(dim, 9, bias=False)
        self.trans = nn.Parameter(torch.rand(9,9,9), requires_grad=True)
        self.norm = nn.LayerNorm(9)
        self.out = nn.Linear(18, 9)
    def forward(self, feat):  # (batch, len, dim)
        last_feat = self.intensity(feat[:,0,:])
        this_feat = self.stimulation(feat[:,1,:])  # (batch, 9)
        batch_list = []
        for i in range(this_feat.shape[0]):
            temp_feat = torch.matmul(last_feat[i], self.trans)  # (9, 9)
            temp_feat = torch.matmul(this_feat[i], temp_feat)  # (9,)
            batch_list.append(temp_feat.unsqueeze(0))
        out_feat = torch.cat(batch_list, dim=0)  # (batch, 9)
        out_feat = torch.cat([this_feat, self.norm(out_feat)], dim=1)
        return self.out(out_feat)

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
        feat, label = zip(*batch)
        feat, label = torch.cuda.FloatTensor(feat), torch.cuda.LongTensor(label)
        logits_clsf = model(feat)
        loss = multi_circle_loss(logits_clsf, label)
        loss = loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)  #梯度裁剪
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
            feat, label = zip(*batch)
            feat, label = torch.cuda.FloatTensor(feat), torch.cuda.LongTensor(label)
            logits_clsf = model(feat)
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
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=6, verbose=True)
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
        if valid_loss == min(loss_list) and valid_loss > 0.009:
            stop = 0
            torch.save(model.state_dict(), os.path.join(log_dir, name+'_'+str(valid_loss)[:4]+'.pt'))
        else:
            stop += 1
            if stop >= 15:
                break
    writer.close()

random.shuffle(train_name_list)
model_1 = Concat_Linear(dim=DIM).to(device)
valid_list = train_name_list[:6720]
train_list = train_name_list[6720:]
run(model_1, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_d1')
model_2 = Concat_Linear(dim=DIM).to(device)
valid_list = train_name_list[6720:13440]
train_list = train_name_list[:6720] + train_name_list[13440:]
run(model_2, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_d2')
model_3 = Concat_Linear(dim=DIM).to(device)
valid_list = train_name_list[13440:20160]
train_list = train_name_list[:13440] + train_name_list[20160:]
run(model_3, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_d3')
model_4 = Concat_Linear(dim=DIM).to(device)
valid_list = train_name_list[20160:26880]
train_list = train_name_list[:20160] + train_name_list[26880:]
run(model_4, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_d4')

model_1 = Concat_Linear(dim=DIM).to(device)
model_1.load_state_dict(torch.load(log_dir + 'model_d1_2.11.pt'))
model_2 = Concat_Linear(dim=DIM).to(device)
model_2.load_state_dict(torch.load(log_dir + 'model_d2_2.12.pt'))
model_3 = Concat_Linear(dim=DIM).to(device)
model_3.load_state_dict(torch.load(log_dir + 'model_d3_2.09.pt'))
model_4 = Concat_Linear(dim=DIM).to(device)
model_4.load_state_dict(torch.load(log_dir + 'model_d4_2.09.pt'))

for name, p in model_1.named_parameters():
    if name == 'trans':
        matrix_1 = nn.Tanh()(p)
for name, p in model_2.named_parameters():
    if name == 'trans':
        matrix_2 = nn.Tanh()(p)
for name, p in model_3.named_parameters():
    if name == 'trans':
        matrix_3 = nn.Tanh()(p)
for name, p in model_4.named_parameters():
    if name == 'trans':
        matrix_4 = nn.Tanh()(p)
print('Transfer matrix:', (matrix_1+matrix_2+matrix_3+matrix_4)/4)

def test(model_1, model_2, model_3, model_4):
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_name_list, batch_size=len(test_name_list))
        for _, batch in enumerate(test_iterator):
            feat, label = zip(*batch)
            feat, label = torch.cuda.FloatTensor(feat), torch.cuda.LongTensor(label)
            pred_1 = (model_1(feat)).detach().cpu()
            pred_2 = (model_2(feat)).detach().cpu()
            pred_3 = (model_3(feat)).detach().cpu()
            pred_4 = (model_4(feat)).detach().cpu()
            pred = (pred_1+pred_2+pred_3+pred_4) / 4
            zero = torch.zeros_like(pred)
            one = torch.ones_like(pred)
            label = label.detach().cpu()
    pred_all, label_all = [], []
    for j in range(label.shape[0]):
        label_all.append(label[j][:8].int().tolist())
    pred_love = torch.where(pred > (-0.7), one, zero)
    pred_anxi = torch.where(pred > (-0.8), one, zero)
    pred_sorr = torch.where(pred > (-0.3), one, zero)
    pred_joyy = torch.where(pred > (-0.2), one, zero)
    pred_expe = torch.where(pred > (-0.2), one, zero)
    pred_hate = torch.where(pred > (-0.8), one, zero)
    pred_ange = torch.where(pred > (-0.8), one, zero)
    pred_surp = torch.where(pred > (-0.9), one, zero)
    for j in range(pred.shape[0]):
        temp_pred = [0,0,0,0,0,0,0,0]
        temp_pred[0] = int(pred_love[j][0])
        temp_pred[1] = int(pred_anxi[j][1])
        temp_pred[2] = int(pred_sorr[j][2])
        temp_pred[3] = int(pred_joyy[j][3])
        temp_pred[4] = int(pred_expe[j][4])
        temp_pred[5] = int(pred_hate[j][5])
        temp_pred[6] = int(pred_ange[j][6])
        temp_pred[7] = int(pred_surp[j][7])
        pred_all.append(temp_pred)
    print('micro_precision: ', precision_score(label_all, pred_all, average='micro'))
    print('micro_recall: ', recall_score(label_all, pred_all, average='micro'))
    print('micro_f1: ', f1_score(label_all, pred_all, average='micro'))
    print('macro_precision: ', precision_score(label_all, pred_all, average='macro'))
    print('macro_recall: ', recall_score(label_all, pred_all, average='macro'))
    print('macro_f1: ', f1_score(label_all, pred_all, average='macro'))
    return 0

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(mat, name):
    plt.imshow(mat, cmap=plt.cm.binary)
    plt.title(name)
    plt.colorbar()
    
    labels = ['Love','Anxiety','Sorrow','Joy','Expect','Hate','Anger','Surprise','Neutral']
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.ylabel('From')
    plt.xlabel('To')
    plt.savefig(log_dir + 'img/' + name)
    plt.show()
    

mat = np.array([[0.3727, 0.5813, 0.4624, 0.4032, 0.3781, 0.2649, 0.3064, 0.3217, 0.4802],
[0.2174, 0.2682, 0.3404, 0.3250, 0.4246, 0.4803, 0.2743, 0.5207, 0.5665],
[0.5619, 0.3931, 0.4475, 0.2598, 0.5106, 0.3409, 0.5110, 0.2609, 0.1741],
[0.3197, 0.4245, 0.2833, 0.6105, 0.4090, 0.3129, 0.2294, 0.3445, 0.3866],
[0.5453, 0.4259, 0.4732, 0.4402, 0.5582, 0.4570, 0.4744, 0.3903, 0.5602],
[0.4340, 0.3873, 0.4853, 0.2537, 0.4865, 0.4162, 0.3143, 0.5525, 0.3492],
[0.4344, 0.3814, 0.4962, 0.4756, 0.5386, 0.4334, 0.3373, 0.3904, 0.3280],
[0.3867, 0.3489, 0.2865, 0.3979, 0.5055, 0.3166, 0.5570, 0.5200, 0.3799],
[0.2155, 0.6568, 0.3900, 0.3049, 0.3733, 0.6049, 0.3878, 0.3091, 0.5865]])
plot_confusion_matrix(mat, 'Love')
