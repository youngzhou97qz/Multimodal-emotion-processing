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
save_dir = '/home/'+user+'/pro_log/'
log_dir = '/home/'+user+'/pro_log/'
L_DIM = 300
V_DIM = 35
A_DIM = 74
DIM = 192
L_LEN = 20
V_LEN = 200
A_LEN = 600
N_HEADS = 12
FFN = 2
N_LAYERS = 1
EPS = 1e-5
BATCH = 8
LR = 0.0001
CLIP = 1.0
EPOCHS = 999

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
            l = data_set.computational_sequences['linguistic'].data[name_list[count]]["features"][:]
            v = data_set.computational_sequences['visual'].data[name_list[count]]["features"][:]
            a = data_set.computational_sequences['acoustic'].data[name_list[count]]["features"][:]
            label = data_set.computational_sequences['label'].data[name_list[count]]["features"][0]
            label = label_processing(label)
            if len(l) >= l_len:
                l_mask = np.ones(l_len)
                l = l[len(l)-l_len:len(l)+1,...]
            else:
                l_mask = np.concatenate((np.ones(len(l)), np.zeros(l_len - len(l))))
                l = np.concatenate([l, np.zeros([l_len]+list(l.shape[1:]))],axis=0)[:l_len,...]
            if len(v) >= v_len:
                v_mask = np.ones(v_len)
                v = v[len(v)-v_len:len(v)+1,...]
            else:
                v_mask = np.concatenate((np.ones(len(v)), np.zeros(v_len - len(v))))
                v = np.concatenate([v, np.zeros([v_len]+list(v.shape[1:]))],axis=0)[:v_len,...]
            if len(a) >= a_len:
                a_mask = np.ones(a_len)
                a = a[len(a)-a_len:len(a)+1,...]
            else:
                a_mask = np.concatenate((np.ones(len(a)), np.zeros(a_len - len(a))))
                a = np.concatenate([a, np.zeros([a_len]+list(a.shape[1:]))],axis=0)[:a_len,...]
            for i in range(len(a)):
                for j in range(len(a[i])):
                    if math.isinf(a[i][j]):
                        a[i][j] = -71.
            batch.append((l, v, a, l_mask, v_mask, a_mask, label))
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
        self.drop = nn.Dropout(0.1)
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

class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.fully_connected = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.scores = None
        self.n_heads = n_heads
    def forward(self, q, k, v, mask):
        '''
        q    → (batch_size, q_len, dim);
        k, v → (batch_size, kv_len, dim);
        mask → (batch_size, kv_len) / (batch_size, q_len, kv_len)
        '''
        q, k, v = self.fully_connected[0](q), self.fully_connected[1](k), self.fully_connected[2](v)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))   
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask[:, None, None, :]
            elif len(mask.shape) == 3:
                mask = torch.unsqueeze(mask, 1)
                mask = mask.repeat(1,self.n_heads,1,1)
            scores -= 10000.0 * (1.0 - mask)
        scores = F.softmax(scores, dim=-1)
        q = (scores @ v).transpose(1, 2).contiguous()
        q = self.merge_last(q, 2)
        self.scores = scores
        return q
    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)
    def merge_last(self, x, n_dims):
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)

class Position_Wise_Feed_Forward(nn.Module):
    def __init__(self, dim, ffn):
        super().__init__()
        self.fully_connected_1 = nn.Linear(dim, dim*ffn)
        self.fully_connected_2 = nn.Linear(dim*ffn, dim)
    def forward(self, x):
        return self.fully_connected_2(nn.ReLU()(self.fully_connected_1(x)))

class Transformer_Blocks(nn.Module):
    def __init__(self, dim, eps, n_heads, n_layers, ffn):
        super().__init__()
        self.normalization = nn.ModuleList([nn.LayerNorm(dim, eps=eps) for _ in range(n_layers*4)])
        self.self_attention = nn.ModuleList([Multi_Head_Self_Attention(dim=dim, n_heads=n_heads) for _ in range(n_layers)])
        self.fully_connected = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.feed_forward = nn.ModuleList([Position_Wise_Feed_Forward(dim=dim, ffn=ffn) for _ in range(n_layers)])
        self.drop = nn.Dropout(0.1)
    def forward(self, q, k, v, mask, layer_num):
        q, k, v = self.normalization[layer_num*4+0](q), self.normalization[layer_num*4+1](k), self.normalization[layer_num*4+2](v)
        q = q + self.drop(self.fully_connected[layer_num](self.self_attention[layer_num](q, k, v, mask)))
        q = q + self.drop(self.feed_forward[layer_num](self.normalization[layer_num*4+3](q)))
        return q

class Multi_class(nn.Module):
    def __init__(self, l_dim, v_dim, a_dim, dim, l_len, v_len, a_len, eps, n_heads, n_layers, ffn):
        super().__init__()
        self.unify_dimension = Unify_Dimension_Conv1d(l_dim, v_dim, a_dim, dim)
        self.linguistic_position = Position_Embedding(l_len, dim)
        self.visual_position = Position_Embedding(v_len, dim)
        self.acoustic_position = Position_Embedding(a_len, dim)
        self.n_layers = n_layers
        self.multimodal_blocks = nn.ModuleList([Transformer_Blocks(dim, eps, n_heads, n_layers, ffn) for _ in range(9)])
        self.transformer_blocks = nn.ModuleList([Transformer_Blocks(dim*3, eps, n_heads, n_layers, ffn) for _ in range(3)])
        self.fully_connected = nn.Linear(dim*6, dim)
        self.normalization = nn.LayerNorm(dim, eps=eps)
        self.drop = nn.Dropout(0.1)
        self.classifier = nn.Linear(dim, 6)
    def forward(self, l, v, a, l_mask, v_mask, a_mask):
        l, v, a = self.unify_dimension(l, v, a)
        l = l + self.linguistic_position(l)
        v = v + self.visual_position(v)
        a = a + self.acoustic_position(a)
        for i in range(self.n_layers):
            ll = self.multimodal_blocks[0](l, l, l, l_mask, i)
            lv = self.multimodal_blocks[1](l, v, v, v_mask, i)
            la = self.multimodal_blocks[2](l, a, a, a_mask, i)
            vv = self.multimodal_blocks[3](v, v, v, v_mask, i)
            vl = self.multimodal_blocks[4](v, l, l, l_mask, i)
            va = self.multimodal_blocks[5](v, a, a, a_mask, i)
            aa = self.multimodal_blocks[6](a, a, a, a_mask, i)
            al = self.multimodal_blocks[7](a, l, l, l_mask, i)
            av = self.multimodal_blocks[8](a, v, v, v_mask, i)
        l = torch.cat([ll, lv, la], dim=2)
        for i in range(self.n_layers):
            l = self.transformer_blocks[0](l, l, l, l_mask, i)
        v = torch.cat([vv, vl, va], dim=2)
        for i in range(self.n_layers):
            v = self.transformer_blocks[1](v, v, v, v_mask, i)
        a = torch.cat([aa, al, av], dim=2)
        for i in range(self.n_layers):
            a = self.transformer_blocks[2](a, a, a, a_mask, i)
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
        linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = zip(*batch)
        linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(label)
        logits_clsf = model(linguistic, visual, acoustic, l_mask, v_mask, a_mask)
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
            linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = zip(*batch)
            linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(label)
            logits_clsf = model(linguistic, visual, acoustic, l_mask, v_mask, a_mask)
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
        train_iterator = data_loader(data_set, train_list, batch_size, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN)
        valid_iterator = data_loader(data_set, valid_list, batch_size, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN)
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

def test(model_1, model_2, model_3, model_4, model_5, iterator, threshold):
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    model_5.eval()
    label_happ, soft_happ, hard_happ = [], [], []
    label_sadn, soft_sadn, hard_sadn = [], [], []
    label_ange, soft_ange, hard_ange = [], [], []
    label_surp, soft_surp, hard_surp = [], [], []
    label_disg, soft_disg, hard_disg = [], [], []
    label_fear, soft_fear, hard_fear = [], [], []
    with torch.no_grad():
        iter_bar = tqdm(iterator, desc='Testing')
        for _, batch in enumerate(iter_bar):
            linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = zip(*batch)
            linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(label)
            pred_1 = (model_1(linguistic, visual, acoustic, l_mask, v_mask, a_mask)).cpu().detach()
            pred_2 = (model_2(linguistic, visual, acoustic, l_mask, v_mask, a_mask)).cpu().detach()
            pred_3 = (model_3(linguistic, visual, acoustic, l_mask, v_mask, a_mask)).cpu().detach()
            pred_4 = (model_4(linguistic, visual, acoustic, l_mask, v_mask, a_mask)).cpu().detach()
            pred_5 = (model_5(linguistic, visual, acoustic, l_mask, v_mask, a_mask)).cpu().detach()
            pred = (pred_1+pred_2+pred_3+pred_4+pred_5)/5
            zero = torch.zeros_like(pred)
            one = torch.ones_like(pred)
            pred_1 = torch.where(pred_1 > threshold, one, zero)
            pred_2 = torch.where(pred_2 > threshold, one, zero)
            pred_3 = torch.where(pred_3 > threshold, one, zero)
            pred_4 = torch.where(pred_4 > threshold, one, zero)
            pred_5 = torch.where(pred_5 > threshold, one, zero)
            pred = torch.where(pred > threshold, one, zero)
            hard_pred = (pred_1+pred_2+pred_3+pred_4+pred_5)/5
            hard_pred = torch.where(hard_pred > 0.5, one, zero)
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
                hard_happ.append(int(hard_pred[i][0]))
                hard_sadn.append(int(hard_pred[i][1]))
                hard_ange.append(int(hard_pred[i][2]))
                hard_surp.append(int(hard_pred[i][3]))
                hard_disg.append(int(hard_pred[i][4]))
                hard_fear.append(int(hard_pred[i][5]))
    happ_soft_acc = accuracy_score(label_happ, soft_happ)
    happ_hard_acc = accuracy_score(label_happ, hard_happ)
    happ_soft_f1 = f1_score(label_happ, soft_happ, average='weighted')
    happ_hard_f1 = f1_score(label_happ, hard_happ, average='weighted')
    sadn_soft_acc = accuracy_score(label_sadn, soft_sadn)
    sadn_hard_acc = accuracy_score(label_sadn, hard_sadn)
    sadn_soft_f1 = f1_score(label_sadn, soft_sadn, average='weighted')
    sadn_hard_f1 = f1_score(label_sadn, hard_sadn, average='weighted')
    ange_soft_acc = accuracy_score(label_ange, soft_ange)
    ange_hard_acc = accuracy_score(label_ange, hard_ange)
    ange_soft_f1 = f1_score(label_ange, soft_ange, average='weighted')
    ange_hard_f1 = f1_score(label_ange, hard_ange, average='weighted')
    surp_soft_acc = accuracy_score(label_surp, soft_surp)
    surp_hard_acc = accuracy_score(label_surp, hard_surp)
    surp_soft_f1 = f1_score(label_surp, soft_surp, average='weighted')
    surp_hard_f1 = f1_score(label_surp, hard_surp, average='weighted')
    disg_soft_acc = accuracy_score(label_disg, soft_disg)
    disg_hard_acc = accuracy_score(label_disg, hard_disg)
    disg_soft_f1 = f1_score(label_disg, soft_disg, average='weighted')
    disg_hard_f1 = f1_score(label_disg, hard_disg, average='weighted')
    fear_soft_acc = accuracy_score(label_fear, soft_fear)
    fear_hard_acc = accuracy_score(label_fear, hard_fear)
    fear_soft_f1 = f1_score(label_fear, soft_fear, average='weighted')
    fear_hard_f1 = f1_score(label_fear, hard_fear, average='weighted')
    return happ_soft_acc, happ_hard_acc, happ_soft_f1, happ_hard_f1, sadn_soft_acc, sadn_hard_acc, sadn_soft_f1, sadn_hard_f1, ange_soft_acc, ange_hard_acc, ange_soft_f1, ange_hard_f1, surp_soft_acc, surp_hard_acc, surp_soft_f1, surp_hard_f1, disg_soft_acc, disg_hard_acc, disg_soft_f1, disg_hard_f1, fear_soft_acc, fear_hard_acc, fear_soft_f1, fear_hard_f1

# random.shuffle(name_list)
model_1 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
valid_list = name_list[:int(len(name_list)*0.2)]
train_list = name_list[int(len(name_list)*0.2):]
run(model_1, data_set, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_1')
model_2 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
valid_list = name_list[int(len(name_list)*0.2):int(len(name_list)*0.4)]
train_list = name_list[:int(len(name_list)*0.2)] + name_list[int(len(name_list)*0.4):]
run(model_2, data_set, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_2')
model_3 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
valid_list = name_list[int(len(name_list)*0.4):int(len(name_list)*0.6)]
train_list = name_list[:int(len(name_list)*0.4)] + name_list[int(len(name_list)*0.6):]
run(model_3, data_set, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_3')
model_4 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
valid_list = name_list[int(len(name_list)*0.6):int(len(name_list)*0.8)]
train_list = name_list[:int(len(name_list)*0.6)] + name_list[int(len(name_list)*0.8):]
run(model_4, data_set, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_4')
model_5 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
valid_list = name_list[int(len(name_list)*0.8):]
train_list = name_list[:int(len(name_list)*0.8)]
run(model_5, data_set, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_5')

model_1 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_1.load_state_dict(torch.load(save_dir + '??.pt'))
model_2 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_2.load_state_dict(torch.load(save_dir + 'model_2_2.05.pt'))
model_3 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_3.load_state_dict(torch.load(save_dir + 'model_3_1.89.pt'))
model_4 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_4.load_state_dict(torch.load(save_dir + 'model_4_1.82.pt'))
model_5 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_5.load_state_dict(torch.load(save_dir + 'model_5_1.83.pt'))
test_iterator = data_loader(data_set, test_list, batch_size=BATCH, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN)

i = 0
happ_soft_acc, happ_hard_acc, happ_soft_f1, happ_hard_f1, sadn_soft_acc, sadn_hard_acc, sadn_soft_f1, sadn_hard_f1, ange_soft_acc, ange_hard_acc, ange_soft_f1, ange_hard_f1, surp_soft_acc, surp_hard_acc, surp_soft_f1, surp_hard_f1, disg_soft_acc, disg_hard_acc, disg_soft_f1, disg_hard_f1, fear_soft_acc, fear_hard_acc, fear_soft_f1, fear_hard_f1 = test(model_1, model_2, model_3, model_4, model_5, test_iterator, i)
print('happ_soft_acc: ', happ_soft_acc)
print('happ_hard_acc: ', happ_hard_acc)
print('happ_soft_f1: ', happ_soft_f1)
print('happ_hard_f1: ', happ_hard_f1)
print('sadn_soft_acc: ', sadn_soft_acc)
print('sadn_hard_acc: ', sadn_hard_acc)
print('sadn_soft_f1: ', sadn_soft_f1)
print('sadn_hard_f1: ', sadn_hard_f1)
print('ange_soft_acc: ', ange_soft_acc)
print('ange_hard_acc: ', ange_hard_acc)
print('ange_soft_f1: ', ange_soft_f1)
print('ange_hard_f1: ', ange_hard_f1)
print('surp_soft_acc: ', surp_soft_acc)
print('surp_hard_acc: ', surp_hard_acc)
print('surp_soft_f1: ', surp_soft_f1)
print('surp_hard_f1: ', surp_hard_f1)
print('disg_soft_acc: ', disg_soft_acc)
print('disg_hard_acc: ', disg_hard_acc)
print('disg_soft_f1: ', disg_soft_f1)
print('disg_hard_f1: ', disg_hard_f1)
print('fear_soft_acc: ', fear_soft_acc)
print('fear_hard_acc: ', fear_hard_acc)
print('fear_soft_f1: ', fear_soft_f1)
print('fear_hard_f1: ', fear_hard_f1)
