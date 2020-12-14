import os
import math
import random
import numpy as np
from tqdm import tqdm
from mmsdk import mmdatasdk
from sklearn.metrics import f1_score, accuracy_score
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
user = 'dango/multimodal/CMU-MOSEI'
data_dir = '/home/'+user+'/align/'
save_dir = '/home/'+user+'/par_log/official_2/'
log_dir = '/home/'+user+'/par_log/official_2/'
L_DIM = 300
V_DIM = 35
A_DIM = 74
LEN = 50
EPS = 1e-5
LR = 0.0001
CLIP = 1.0
DROP = 0.1
EPOCHS = 99
BATCH = 128  # 4 8 16 32
DIM = 96  # 96 192 288 384
N_HEADS = 6  # 6 8 12 16
FFN = 2  # 1 2 3 4
N_LAYERS = 2 # 1 2 3 4

# data
data_dict={'linguistic':data_dir+'glove_vectors.csd', 
       'acoustic':data_dir+'COAVAREP.csd', 
       'visual':data_dir+'FACET 4.2.csd', 
       'label':data_dir+'All Labels.csd'}
data_set=mmdatasdk.mmdataset(data_dict)
name_list, test_list = [], []
for name in data_set.computational_sequences['visual'].data.keys():
    if name.split('[')[0] in mmdatasdk.cmu_mosei.standard_folds.standard_train_fold or name.split('[')[0] in mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold:
        name_list.append(name)
    elif name.split('[')[0] in mmdatasdk.cmu_mosei.standard_folds.standard_test_fold:
        test_list.append(name)

def label_processing(l):
    label = l[1:]
    label[0] = 1 if label[0] > 0 else 0
    label[1] = 1 if label[1] > 0 else 0
    label[2] = 1 if label[2] > 0 else 0
    label[3] = 1 if label[3] > 0 else 0
    label[4] = 1 if label[4] > 0 else 0
    label[5] = 1 if label[5] > 0 else 0
    return label

train_set, test_set = {}, {}
for i in tqdm(range(len(name_list)), desc='Loading train_set...'):
    l = data_set.computational_sequences['linguistic'].data[name_list[i]]["features"][:]
    v = data_set.computational_sequences['visual'].data[name_list[i]]["features"][:]
    a = data_set.computational_sequences['acoustic'].data[name_list[i]]["features"][:]
    if len(l) >= LEN:
        l = l[len(l)-LEN:len(l)+1,...]
    else:
        l = np.concatenate([l, np.zeros([LEN]+list(l.shape[1:]))],axis=0)[:LEN,...]
    if len(v) >= LEN:
        v = v[len(v)-LEN:len(v)+1,...]
    else:
        v = np.concatenate([v, np.zeros([LEN]+list(v.shape[1:]))],axis=0)[:LEN,...]
    if len(a) >= LEN:
        a = a[len(a)-LEN:len(a)+1,...]
    else:
        a = np.concatenate([a, np.zeros([LEN]+list(a.shape[1:]))],axis=0)[:LEN,...]
    for j in range(len(a)):
        for k in range(len(a[j])):
            if math.isinf(a[j][k]) or math.isnan(a[j][k]):
                a[j][k] = -71.
    train_set[name_list[i]] = {'linguistic':l, 'visual':v, 'acoustic':a, 'label':label_processing(data_set.computational_sequences['label'].data[name_list[i]]["features"][0])}

for i in tqdm(range(len(test_list)), desc='Loading test_set...'):
    l = data_set.computational_sequences['linguistic'].data[test_list[i]]["features"][:]
    v = data_set.computational_sequences['visual'].data[test_list[i]]["features"][:]
    a = data_set.computational_sequences['acoustic'].data[test_list[i]]["features"][:]
    if len(l) >= LEN:
        l = l[len(l)-LEN:len(l)+1,...]
    else:
        l = np.concatenate([l, np.zeros([LEN]+list(l.shape[1:]))],axis=0)[:LEN,...]
    if len(v) >= LEN:
        v = v[len(v)-LEN:len(v)+1,...]
    else:
        v = np.concatenate([v, np.zeros([LEN]+list(v.shape[1:]))],axis=0)[:LEN,...]
    if len(a) >= LEN:
        a = a[len(a)-LEN:len(a)+1,...]
    else:
        a = np.concatenate([a, np.zeros([LEN]+list(a.shape[1:]))],axis=0)[:LEN,...]
    for j in range(len(a)):
        for k in range(len(a[j])):
            if math.isinf(a[j][k]) or math.isnan(a[j][k]):
                a[j][k] = -71.
    test_set[test_list[i]] = {'linguistic':l, 'visual':v, 'acoustic':a, 'label':label_processing(data_set.computational_sequences['label'].data[test_list[i]]["features"][0])}

# balance
num_happ, num_sadn, num_ange, num_fear, num_disg, num_surp = 0, 0, 0, 0, 0, 0  # 9740 4789 3864 1845 3236 1507
for i in range(len(name_list)):
    if train_set[name_list[i]]['label'][0] == 1:
        num_happ += 1
    if train_set[name_list[i]]['label'][1] == 1:
        num_sadn += 1
    if train_set[name_list[i]]['label'][2] == 1:
        num_ange += 1
    if train_set[name_list[i]]['label'][3] == 1:
        num_fear += 1
    if train_set[name_list[i]]['label'][4] == 1:
        num_disg += 1
    if train_set[name_list[i]]['label'][5] == 1:
        num_surp += 1

num_011111 = num_happ - num_sadn
num_001111 = num_sadn - num_ange
num_000111 = num_ange - num_disg
num_000101 = num_disg - num_fear
num_000001 = num_fear - num_surp
list_011111 = []
list_001111 = []
list_000111 = []
list_000101 = []
list_000001 = []  # 19 4 6 22 187
for i in range(len(name_list)):
    if (train_set[name_list[i]]['label'] == np.array([0,1,1,1,1,1])).all():
        list_011111.append(name_list[i])
    if (train_set[name_list[i]]['label'] == np.array([0,0,1,1,1,1])).all():
        list_001111.append(name_list[i])
    if (train_set[name_list[i]]['label'] == np.array([0,0,0,1,1,1])).all():
        list_000111.append(name_list[i])
    if (train_set[name_list[i]]['label'] == np.array([0,0,0,1,0,1])).all():
        list_000101.append(name_list[i])
    if (train_set[name_list[i]]['label'] == np.array([0,0,0,0,0,1])).all():
        list_000001.append(name_list[i])

for i in range(num_011111):
    temp = random.sample(list_011111, 2)
    alpha = random.random()
    name_list.append('num_011111'+str(i))
    train_set['num_011111'+str(i)] = {'linguistic':alpha*train_set[temp[0]]['linguistic'] + (1-alpha)*train_set[temp[1]]['linguistic'],\
                               'acoustic':alpha*train_set[temp[0]]['acoustic'] + (1-alpha)*train_set[temp[1]]['acoustic'],\
                               'visual':alpha*train_set[temp[0]]['visual'] + (1-alpha)*train_set[temp[1]]['visual'],\
                               'label':np.array([0,1,1,1,1,1])}
for i in range(num_001111):
    temp = random.sample(list_001111, 2)
    alpha = random.random()
    name_list.append('num_001111'+str(i))
    train_set['num_001111'+str(i)] = {'linguistic':alpha*train_set[temp[0]]['linguistic'] + (1-alpha)*train_set[temp[1]]['linguistic'],\
                               'acoustic':alpha*train_set[temp[0]]['acoustic'] + (1-alpha)*train_set[temp[1]]['acoustic'],\
                               'visual':alpha*train_set[temp[0]]['visual'] + (1-alpha)*train_set[temp[1]]['visual'],\
                               'label':np.array([0,0,1,1,1,1])}
for i in range(num_000111):
    temp = random.sample(list_000111, 2)
    alpha = random.random()
    name_list.append('num_000111'+str(i))
    train_set['num_000111'+str(i)] = {'linguistic':alpha*train_set[temp[0]]['linguistic'] + (1-alpha)*train_set[temp[1]]['linguistic'],\
                               'acoustic':alpha*train_set[temp[0]]['acoustic'] + (1-alpha)*train_set[temp[1]]['acoustic'],\
                               'visual':alpha*train_set[temp[0]]['visual'] + (1-alpha)*train_set[temp[1]]['visual'],\
                               'label':np.array([0,0,0,1,1,1])}
for i in range(num_000101):
    temp = random.sample(list_000101, 2)
    alpha = random.random()
    name_list.append('num_000101'+str(i))
    train_set['num_000101'+str(i)] = {'linguistic':alpha*train_set[temp[0]]['linguistic'] + (1-alpha)*train_set[temp[1]]['linguistic'],\
                               'acoustic':alpha*train_set[temp[0]]['acoustic'] + (1-alpha)*train_set[temp[1]]['acoustic'],\
                               'visual':alpha*train_set[temp[0]]['visual'] + (1-alpha)*train_set[temp[1]]['visual'],\
                               'label':np.array([0,0,0,1,0,1])}
for i in range(num_000001):
    temp = random.sample(list_000001, 2)
    alpha = random.random()
    name_list.append('num_000001'+str(i))
    train_set['num_000001'+str(i)] = {'linguistic':alpha*train_set[temp[0]]['linguistic'] + (1-alpha)*train_set[temp[1]]['linguistic'],\
                               'acoustic':alpha*train_set[temp[0]]['acoustic'] + (1-alpha)*train_set[temp[1]]['acoustic'],\
                               'visual':alpha*train_set[temp[0]]['visual'] + (1-alpha)*train_set[temp[1]]['visual'],\
                               'label':np.array([0,0,0,0,0,1])}
random.shuffle(name_list)

def data_loader(data_set, name_list, batch_size, l_len, v_len, a_len):
    random.shuffle(name_list)
    count = 0
    while count < len(name_list):
        batch = []
        if batch_size < len(name_list) - count:
            size = batch_size
        else: 
            size = len(name_list) - count
        for _ in range(size):
            l = data_set[name_list[count]]['linguistic']
            v = data_set[name_list[count]]['visual']
            a = data_set[name_list[count]]['acoustic']
            label = data_set[name_list[count]]['label']
            batch.append((l, v, a, label, name_list[count]))
            count += 1
        yield batch

# model
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class Unify_Dimension_Conv1d(nn.Module):
    def __init__(self, l_dim, v_dim, a_dim, dim):
        super().__init__()
        self.linguistic = nn.Conv1d(l_dim, dim, kernel_size=1, bias=False)
        self.visual = nn.Conv1d(v_dim, dim, kernel_size=1, bias=False)
        self.acoustic = nn.Conv1d(a_dim, dim, kernel_size=1, bias=False)
        self.drop = nn.Dropout(DROP)
    def forward(self, l, v, a):
        l, v, a = l.transpose(1, 2), v.transpose(1, 2), a.transpose(1, 2)
        l, v, a = self.drop(self.linguistic(l)), self.drop(self.visual(v)), self.drop(self.acoustic(a))
        return l.transpose(1, 2), v.transpose(1, 2), a.transpose(1, 2)

class Position_Embedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, dim)
        self.len = max_len
    def forward(self, x):
        position_ids = torch.arange(self.len, device=device).unsqueeze(0).repeat(x.size()[0],1)
        return self.position_embeddings(position_ids.to(device))

class Multi_class(nn.Module):
    def __init__(self, l_dim, v_dim, a_dim, dim, l_len, v_len, a_len, eps, n_heads, n_layers, ffn):
        super().__init__()
        self.unify_dimension = Unify_Dimension_Conv1d(l_dim, v_dim, a_dim, dim)
        self.linguistic_position = Position_Embedding(l_len, dim)
        self.visual_position = Position_Embedding(v_len, dim)
        self.acoustic_position = Position_Embedding(a_len, dim)
        self.multi_1 = nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*ffn, dropout=DROP)
        self.multi_blocks_1 = nn.TransformerDecoder(self.multi_1, num_layers=n_layers)
        self.multi_2 = nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*ffn, dropout=DROP)
        self.multi_blocks_2 = nn.TransformerDecoder(self.multi_2, num_layers=n_layers)
        self.multi_3 = nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*ffn, dropout=DROP)
        self.multi_blocks_3 = nn.TransformerDecoder(self.multi_3, num_layers=n_layers)
        self.multi_4 = nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*ffn, dropout=DROP)
        self.multi_blocks_4 = nn.TransformerDecoder(self.multi_4, num_layers=n_layers)
        self.multi_5 = nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*ffn, dropout=DROP)
        self.multi_blocks_5 = nn.TransformerDecoder(self.multi_5, num_layers=n_layers)
        self.multi_6 = nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*ffn, dropout=DROP)
        self.multi_blocks_6 = nn.TransformerDecoder(self.multi_6, num_layers=n_layers)
        self.multi_7 = nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*ffn, dropout=DROP)
        self.multi_blocks_7 = nn.TransformerDecoder(self.multi_7, num_layers=n_layers)
        self.multi_8 = nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*ffn, dropout=DROP)
        self.multi_blocks_8 = nn.TransformerDecoder(self.multi_8, num_layers=n_layers)
        self.multi_9 = nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*ffn, dropout=DROP)
        self.multi_blocks_9 = nn.TransformerDecoder(self.multi_9, num_layers=n_layers)
        self.mono_1 = nn.TransformerEncoderLayer(d_model=dim*3, nhead=n_heads, dim_feedforward=dim*ffn*3, dropout=DROP)
        self.mono_blocks_1 = nn.TransformerEncoder(self.mono_1, num_layers=n_layers)
        self.mono_2 = nn.TransformerEncoderLayer(d_model=dim*3, nhead=n_heads, dim_feedforward=dim*ffn*3, dropout=DROP)
        self.mono_blocks_2 = nn.TransformerEncoder(self.mono_2, num_layers=n_layers)
        self.mono_3 = nn.TransformerEncoderLayer(d_model=dim*3, nhead=n_heads, dim_feedforward=dim*ffn*3, dropout=DROP)
        self.mono_blocks_3 = nn.TransformerEncoder(self.mono_3, num_layers=n_layers)
        self.fully_connected = nn.Linear(dim*6, dim)
        self.normalization = nn.LayerNorm(dim, eps=eps)
        self.drop = nn.Dropout(DROP)
        self.classifier = nn.Linear(dim, 6)
    def forward(self, l, v, a):
        l, v, a = self.unify_dimension(l, v, a)
        l = l + self.linguistic_position(l)
        v = v + self.visual_position(v)
        a = a + self.acoustic_position(a)
        l, v, a = l.transpose(0, 1), v.transpose(0, 1), a.transpose(0, 1)
        ll = self.multi_blocks_1(l, l)
        lv = self.multi_blocks_2(l, v)
        la = self.multi_blocks_3(l, a)
        vv = self.multi_blocks_4(v, v)
        vl = self.multi_blocks_5(v, l)
        va = self.multi_blocks_6(v, a)
        aa = self.multi_blocks_7(a, a)
        al = self.multi_blocks_8(a, l)
        av = self.multi_blocks_9(a, v)
        l = torch.cat([ll, lv, la], dim=2)
        l = self.mono_blocks_1(l)
        v = torch.cat([vv, vl, va], dim=2)
        v = self.mono_blocks_2(v)
        a = torch.cat([aa, al, av], dim=2)
        a = self.mono_blocks_2(a)
        l, v, a = l.transpose(0, 1), v.transpose(0, 1), a.transpose(0, 1)
        x = torch.cat([l, a, v], dim=1)  # (batch, l_len+v_len+a_len, dim*3)
        x = torch.cat([torch.mean(x, 1), torch.max(x, 1)[0]], dim=1)  # (batch, dim*6)
        x = self.drop(nn.ReLU()(self.normalization(self.fully_connected(x))))
        return self.classifier(x)

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
        linguistic, visual, acoustic, label, name= zip(*batch)
        linguistic, visual, acoustic, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.LongTensor(label)
        logits_clsf = model(linguistic, visual, acoustic)
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
            linguistic, visual, acoustic, label, name = zip(*batch)
            linguistic, visual, acoustic, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.LongTensor(label)
            logits_clsf = model(linguistic, visual, acoustic)
            loss = multi_circle_loss(logits_clsf, label)
            loss = loss.mean()
            iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss, count, epoch_loss / count

def run(model, data_set, train_list, valid_list, batch_size, learning_rate, epochs, name):
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
        train_iterator = data_loader(data_set, train_list, batch_size, l_len=LEN, v_len=LEN, a_len=LEN)
        valid_iterator = data_loader(data_set, valid_list, batch_size, l_len=LEN, v_len=LEN, a_len=LEN)
        train_loss = train(model, train_iterator, optimizer)
        _, _, valid_loss = valid(model, valid_iterator)
        writer.add_scalars(name, {'train_loss':train_loss, 'valid_loss':valid_loss}, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss) 
        with open(log_file, 'a') as log_f:
            log_f.write('\n{epoch},{train_loss: 2.2f},{valid_loss: 2.2f}\n'.format(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss))
        if valid_loss == min(loss_list):
            stop = 0
            torch.save(model.state_dict(), os.path.join(save_dir, name+'_'+str(valid_loss)[:4]+'.pt'))
        else:
            stop += 1
            if stop >= 4:
                break
    writer.close()

def test(model_1, model_2, iterator, threshold):
    model_1.eval()
    model_2.eval()
    label_happ, soft_happ = [], []
    label_sadn, soft_sadn = [], []
    label_ange, soft_ange = [], []
    label_surp, soft_surp = [], []
    label_disg, soft_disg = [], []
    label_fear, soft_fear = [], []
    with torch.no_grad():
        iter_bar = tqdm(iterator, desc='Testing')
        for _, batch in enumerate(iter_bar):
            linguistic, visual, acoustic, label, name = zip(*batch)
            linguistic, visual, acoustic, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.LongTensor(label)
            pred_1 = (model_1(linguistic, visual, acoustic)).cpu().detach()
            pred_2 = (model_2(linguistic, visual, acoustic)).cpu().detach()
            pred = pred_1 * 0.6 + pred_2 * 0.4
            zero = torch.zeros_like(pred)
            one = torch.ones_like(pred)
            pred = torch.where(pred > threshold, one, zero)
            label = label.cpu().detach()
            for i in range(len(label)):
                label_happ.append(int(label[i][0]))
                label_sadn.append(int(label[i][1]))
                label_ange.append(int(label[i][2]))
                label_surp.append(int(label[i][3]))
                label_disg.append(int(label[i][4]))
                label_fear.append(int(label[i][5]))
                soft_happ.append(int(pred[i][0]))
                soft_sadn.append(int(pred[i][1]))
                soft_ange.append(int(pred[i][2]))
                soft_surp.append(int(pred[i][3]))
                soft_disg.append(int(pred[i][4]))
                soft_fear.append(int(pred[i][5]))
    happ_acc = accuracy_score(label_happ, soft_happ)
    happ_f1 = f1_score(label_happ, soft_happ, average='weighted')
    sadn_acc = accuracy_score(label_sadn, soft_sadn)
    sadn_f1 = f1_score(label_sadn, soft_sadn, average='weighted')
    ange_acc = accuracy_score(label_ange, soft_ange)
    ange_f1 = f1_score(label_ange, soft_ange, average='weighted')
    surp_acc = accuracy_score(label_surp, soft_surp)
    surp_f1 = f1_score(label_surp, soft_surp, average='weighted')
    disg_acc = accuracy_score(label_disg, soft_disg)
    disg_f1 = f1_score(label_disg, soft_disg, average='weighted')
    fear_acc = accuracy_score(label_fear, soft_fear)
    fear_f1 = f1_score(label_fear, soft_fear, average='weighted')
    return happ_acc, happ_f1, sadn_acc, sadn_f1, ange_acc, ange_f1, surp_acc, surp_f1, disg_acc, disg_f1, fear_acc, fear_f1

model_1 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=LEN, v_len=LEN, a_len=LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
valid_list = name_list[:int(len(name_list)*0.2)]
train_list = name_list[int(len(name_list)*0.2):]
run(model_1, train_set, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_1')
model_2 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=LEN, v_len=LEN, a_len=LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
valid_list = name_list[int(len(name_list)*0.2):int(len(name_list)*0.4)]
train_list = name_list[:int(len(name_list)*0.2)] + name_list[int(len(name_list)*0.4):]
run(model_2, train_set, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_2')

model_1 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=LEN, v_len=LEN, a_len=LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_1.load_state_dict(torch.load(save_dir + 'model_1_1.33.pt'))
model_2 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=LEN, v_len=LEN, a_len=LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_2.load_state_dict(torch.load(save_dir + 'model_2_1.36.pt'))

test_iterator = data_loader(test_set, test_list, batch_size=BATCH, l_len=LEN, v_len=LEN, a_len=LEN)
th = 0.5
happ_acc, happ_f1, sadn_acc, sadn_f1, ange_acc, ange_f1, surp_acc, surp_f1, disg_acc, disg_f1, fear_acc, fear_f1 = test(model_1, model_2, test_iterator, th)
print(th, happ_f1*0.5373230373230373+sadn_f1*0.24217074217074216+ange_f1*0.22972972972972974+surp_f1*0.0945945945945946+disg_f1*0.17267267267267267+fear_f1*0.08258258258258258)
print('happ_acc: ', happ_acc)
print('happ_f1: ', happ_f1)
print('sadn_acc: ', sadn_acc)
print('sadn_f1: ', sadn_f1)
print('ange_acc: ', ange_acc)
print('ange_f1: ', ange_f1)
print('surp_acc: ', surp_acc)
print('surp_f1: ', surp_f1)
print('disg_acc: ', disg_acc)
print('disg_f1: ', disg_f1)
print('fear_acc: ', fear_acc)
print('fear_f1: ', fear_f1)
