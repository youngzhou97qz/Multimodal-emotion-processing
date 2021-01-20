import os
import math
import random
import numpy as np
from tqdm import tqdm
from mmsdk import mmdatasdk
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
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
log_dir = '/home/'+user+'/par_log/P6_fa_m1_d0_l3_h12_d96_f2/'
EPOCHS = 99
CLIP = 1.0
LR = 0.01
BATCH = 32
L_DIM = 300
V_DIM = 35
A_DIM = 74
L_LEN = 50
V_LEN = 50
A_LEN = 50
FFN = 2  #  4中
DIM = 96  #  192大
N_HEADS = 12  #  6小
N_LAYERS = 3
DROP = 0.0  #  0.2小
Miu = 1  #  3小
AUG = False  #  True中
P_LEN = 6  #  10中

# data
data_dict = {'linguistic':data_dir+'glove_vectors.csd', 'acoustic':data_dir+'COAVAREP.csd', 'visual':data_dir+'FACET 4.2.csd', 'label':data_dir+'All Labels.csd'}
data_set = mmdatasdk.mmdataset(data_dict)

train_name, test_name = [], []
for name in data_set.computational_sequences['visual'].data.keys():
    if name.split('[')[0] in mmdatasdk.cmu_mosei.standard_folds.standard_test_fold:
        test_name.append(name.split('[')[0])
    else:
        train_name.append(name.split('[')[0])
train_name, test_name = set(train_name), set(test_name)

def name_list(name):
    name_list = []
    for n in name:
        temp_list = []
        count = 0
        for i in range((98//P_LEN+1)*P_LEN):
            if n+'['+str(i)+']' in data_set.computational_sequences['visual'].data.keys():
                temp_list.append(n+'['+str(i)+']')
            else:
                temp_list.append('no_name')
            count += 1
            if count == P_LEN:
                count = 0
                if temp_list[0] != 'no_name':
                    name_list.append(temp_list)
                temp_list = []
    return name_list
train_name_list = name_list(train_name)
test_name_list = name_list(test_name)

def masking(m, m_len, is_audio=False):
    m_max = m.max(axis=0)
    m_min = m.min(axis=0)
    m_mean = m.mean(axis=0)
    if len(m) >= m_len-3:
        m_mask = np.ones(m_len)
        m = m[len(m)//2-(m_len-3)//2:len(m)//2-(m_len-3)//2+m_len-3, ...]
        m = np.concatenate((np.expand_dims(m_max, axis=0), np.expand_dims(m_min, axis=0), np.expand_dims(m_mean, axis=0), m), axis=0)
    else:
        m_mask = np.concatenate((np.ones(len(m)+3), np.zeros(m_len - len(m)-3)))
        m = np.concatenate((np.expand_dims(m_max, axis=0), np.expand_dims(m_min, axis=0), np.expand_dims(m_mean, axis=0), m), axis=0)
        m = np.concatenate([m, np.zeros([m_len]+list(m.shape[1:]))],axis=0)[:m_len,...]
    if is_audio:
        for i in range(len(m)):
            for j in range(len(m[i])):
                if math.isinf(m[i][j]) or math.isnan(m[i][j]):
                    m[i][j] = -70.
    return m, m_mask

def label_processing(l):  # sent happ sadn ange surp disg fear
    label = [0, 0, 0, 0, 0, 0, 0]  # happ sadn ange disg surp fear neut
    label[0] = 1 if l[1] > 0 else 0
    label[1] = 1 if l[2] > 0 else 0
    label[2] = 1 if l[3] > 0 else 0
    label[3] = 1 if l[5] > 0 else 0
    label[4] = 1 if l[4] > 0 else 0
    label[5] = 1 if l[6] > 0 else 0
    if sum(label[:6]) == 0:
        label[6] = 1
    return label+label

def augmentation(train_list):
    augment_list = []
    augment_dict = {}
    for li in train_list:
        temp_np = np.array([0, 0, 0, 0, 0, 0, 0])
        count = 0
        for na in li:
            if na != 'no_name':
                if label_processing(data_set.computational_sequences['label'].data[na]["features"][0])[5] != 0 or label_processing(data_set.computational_sequences['label'].data[na]["features"][0])[4] != 0:
                    if label_processing(data_set.computational_sequences['label'].data[na]["features"][0])[0] == 0:
                        if label_processing(data_set.computational_sequences['label'].data[na]["features"][0])[6] == 0:
                            augment_dict[na] = label_processing(data_set.computational_sequences['label'].data[na]["features"][0])[:7]
    name_list = list(augment_dict.keys())
    random.shuffle(name_list)
    for nam in name_list:
        if nam in augment_dict.keys():
            for j in range(len(name_list)):
                if name_list[j] != nam and name_list[j] in augment_dict.keys():
                    augment_list.append([nam, name_list[j]])
                    del augment_dict[nam]
                    del augment_dict[name_list[j]]
                    break
    return augment_list

def kernel_bias(modal='linguistic', _dim=L_DIM, _len=L_LEN, is_audio=False):
    temp_list = []
    for name in tqdm(data_set.computational_sequences[modal].data.keys()):
        temp, _ = masking(data_set.computational_sequences[modal].data[name]["features"][-_len:], _len, is_audio)
        temp_list.append(temp)
    temp = np.concatenate(temp_list, axis=0)
    bias = temp.mean(axis=0, keepdims=True).squeeze()
    cov = np.cov(temp.T)
    u, s, vh = np.linalg.svd(cov)
    kernel = np.dot(u, np.diag(s**0.5))+1e-12*np.random.rand(_dim, _dim)
    kernel = np.linalg.inv(kernel.T)
    return kernel, -bias
l_k, l_b = kernel_bias(modal='linguistic', _dim=L_DIM, _len=L_LEN)
v_k, v_b = kernel_bias(modal='visual', _dim=V_DIM, _len=V_LEN)
a_k, a_b = kernel_bias(modal='acoustic', _dim=A_DIM, _len=A_LEN, is_audio=True)

def whitening(vec, kernel=None, bias=None):
    if not (kernel is None or bias is None):
        vec = (vec + bias).dot(kernel)
    return vec / (vec**2).sum(axis=1, keepdims=True)**0.5

def data_loader(data_set, name_list, batch_size, augment_list=None):  # (batch, y_len, x_len, x_dim) → (batch, y_len*2, 7)
    random.shuffle(name_list)
    count = 0
    while count < len(name_list):
        batch = []
        size = min(batch_size, len(name_list) - count)
        for _ in range(size):
            l_temp, v_temp, a_temp, lmask_temp, vmask_temp, amask_temp, label_temp, mask_temp = [], [], [], [], [], [], [], []
            for i in range(P_LEN):
                if name_list[count][i] != 'no_name':
                    l, l_mask = masking(data_set.computational_sequences['linguistic'].data[name_list[count][i]]["features"][-L_LEN:], L_LEN)
                    v, v_mask = masking(data_set.computational_sequences['visual'].data[name_list[count][i]]["features"][-V_LEN:], V_LEN)
                    a, a_mask = masking(data_set.computational_sequences['acoustic'].data[name_list[count][i]]["features"][-A_LEN:], A_LEN, is_audio=True)
                    l = whitening(l, kernel=l_k, bias=l_b)
                    v = whitening(v, kernel=v_k, bias=v_b)
                    a = whitening(a, kernel=a_k, bias=a_b)
                    label = label_processing(data_set.computational_sequences['label'].data[name_list[count][i]]["features"][0])
                    mask = 1
                elif augment_list != None:
                    alpha = random.random()
                    choice_list = random.choice(augment_list)
                    name_0, name_1 = choice_list[0], choice_list[1]
                    t_len = min(data_set.computational_sequences['linguistic'].data[name_0]["features"].shape[0], data_set.computational_sequences['linguistic'].data[name_1]["features"].shape[0])
                    l = alpha * data_set.computational_sequences['linguistic'].data[name_0]["features"][-t_len:] + (1-alpha) * data_set.computational_sequences['linguistic'].data[name_1]["features"][-t_len:]
                    t_len = min(data_set.computational_sequences['visual'].data[name_0]["features"].shape[0], data_set.computational_sequences['visual'].data[name_1]["features"].shape[0])
                    v = alpha * data_set.computational_sequences['visual'].data[name_0]["features"][-t_len:] + (1-alpha) * data_set.computational_sequences['visual'].data[name_1]["features"][-t_len:]
                    t_len = min(data_set.computational_sequences['acoustic'].data[name_0]["features"].shape[0], data_set.computational_sequences['acoustic'].data[name_1]["features"].shape[0])
                    a = alpha * data_set.computational_sequences['acoustic'].data[name_0]["features"][-t_len:] + (1-alpha) * data_set.computational_sequences['acoustic'].data[name_1]["features"][-t_len:]
                    l, l_mask = masking(l, L_LEN)
                    v, v_mask = masking(v, V_LEN)
                    a, a_mask = masking(a, A_LEN, is_audio=True)
                    l = whitening(l, kernel=l_k, bias=l_b)
                    v = whitening(v, kernel=v_k, bias=v_b)
                    a = whitening(a, kernel=a_k, bias=a_b)
                    label = label_processing(data_set.computational_sequences['label'].data[name_0]["features"][0])
                    mask = 1
                else:
                    l, v, a = np.zeros((L_LEN, L_DIM)), np.zeros((V_LEN, V_DIM)), np.zeros((A_LEN, A_DIM))
                    l_mask, v_mask, a_mask = np.zeros(L_LEN), np.zeros(V_LEN), np.zeros(A_LEN)
                    label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    mask = 0
                l_temp.append(np.expand_dims(l, axis=0))
                v_temp.append(np.expand_dims(v, axis=0))
                a_temp.append(np.expand_dims(a, axis=0))
                lmask_temp.append(np.expand_dims(l_mask, axis=0))
                vmask_temp.append(np.expand_dims(v_mask, axis=0))
                amask_temp.append(np.expand_dims(a_mask, axis=0))
                label_temp.append(np.expand_dims(label, axis=0))
                mask_temp.append(mask)
            batch.append((np.concatenate(l_temp, axis=0), np.concatenate(v_temp, axis=0), np.concatenate(a_temp, axis=0), np.concatenate(label_temp, axis=0),\
                          np.concatenate(lmask_temp, axis=0), np.concatenate(vmask_temp, axis=0), np.concatenate(amask_temp, axis=0), np.asarray(mask_temp)))
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

class Attention_Block(nn.Module):
    def __init__(self, dim, n_heads, ffn):
        super().__init__()
        self.w_qkv = nn.ModuleList([nn.Linear(dim, dim, bias = False) for _ in range(3)])
        self.n_heads = n_heads
        self.drop = nn.Dropout(DROP)
        self.proj = nn.Linear(dim, dim, bias = False)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn * dim), nn.ReLU(), nn.Linear(ffn * dim, dim), nn.Dropout(DROP))
        self.a = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.b = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.c = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
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
    def multi_head_attention(self, q, k, v, mask, scores = None):
        '''
        q    → (batch_size, q_len, dim);
        k, v → (batch_size, kv_len, dim);
        mask → (batch_size, kv_len) / (batch_size, q_len, kv_len)
        '''
        q, k, v = self.w_qkv[0](q), self.w_qkv[1](k), self.w_qkv[2](v)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        if scores is not None: 
            scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1)) + self.c * scores
        else:
            scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask[:, None, None, :]
            elif len(mask.shape) == 3:
                mask = torch.unsqueeze(mask, 1)
                mask = mask.repeat(1,self.n_heads,1,1)
            scores -= 1.0e8 * (1.0 - mask)
        att = F.softmax(scores, dim=-1)
        q = (att @ v).transpose(1, 2).contiguous()
        q = self.merge_last(q, 2)
        return self.drop(self.proj(q)), scores
    def forward(self, q, k, v, mask, scores = None):
        x, scores =  self.multi_head_attention(q, k, v, mask, scores)
        q = self.norm1(q + self.a * x)
        q = self.norm2(q + self.b * self.ffn(q))
        return q, scores

class Multi_class(nn.Module):
    def __init__(self, l_dim, v_dim, a_dim, dim, l_len, v_len, a_len, n_heads, n_layers, ffn):
        super().__init__()
        self.unify_dimension = Unify_Dimension_Conv1d(l_dim, v_dim, a_dim, dim)
        self.linguistic_position = Position_Embedding(l_len, dim)
        self.visual_position = Position_Embedding(v_len, dim)
        self.acoustic_position = Position_Embedding(a_len, dim)
        self.n_layers = n_layers
        self.multimodal_blocks = nn.ModuleList([Attention_Block(dim, n_heads, ffn) for _ in range(9*n_layers)])
        self.fully_connected = nn.Linear(dim*6, dim)
        self.normalization = nn.LayerNorm(dim)
        self.drop = nn.Dropout(DROP)
    def forward(self, l, v, a, l_mask, v_mask, a_mask):
        l, v, a = self.unify_dimension(l, v, a)
        l = l + self.linguistic_position(l)
        v = v + self.visual_position(v)
        a = a + self.acoustic_position(a)
        ll, lv, la = l, l, l
        vv, vl, va = v, v, v
        aa, al, av = a, a, a
        scores = None
        for i in range(self.n_layers):
            ll, scores = self.multimodal_blocks[self.n_layers*0+i](ll, l, l, l_mask, scores)
        scores = None
        for i in range(self.n_layers):
            lv, scores = self.multimodal_blocks[self.n_layers*1+i](lv, v, v, v_mask, scores)
        scores = None
        for i in range(self.n_layers):
            la, scores = self.multimodal_blocks[self.n_layers*2+i](la, a, a, a_mask, scores)
        scores = None
        for i in range(self.n_layers):
            vv, scores = self.multimodal_blocks[self.n_layers*3+i](vv, v, v, v_mask, scores)
        scores = None
        for i in range(self.n_layers):
            vl, scores = self.multimodal_blocks[self.n_layers*4+i](vl, l, l, l_mask, scores)
        scores = None
        for i in range(self.n_layers):
            va, scores = self.multimodal_blocks[self.n_layers*5+i](va, a, a, a_mask, scores)
        scores = None
        for i in range(self.n_layers):
            aa, scores = self.multimodal_blocks[self.n_layers*6+i](aa, a, a, a_mask, scores)
        scores = None
        for i in range(self.n_layers):
            al, scores = self.multimodal_blocks[self.n_layers*7+i](al, l, l, l_mask, scores)
        scores = None
        for i in range(self.n_layers):
            av, scores = self.multimodal_blocks[self.n_layers*8+i](av, v, v, v_mask, scores)
        l = torch.cat([ll, lv, la], dim=2)
        v = torch.cat([vv, vl, va], dim=2)
        a = torch.cat([aa, al, av], dim=2)
        x = torch.cat([l, a, v], dim=1)  # (batch, l_len+v_len+a_len, dim*3)
        x = torch.cat([torch.mean(x, 1), torch.max(x, 1)[0]], dim=1)  # (batch, dim*6)
#         x = self.drop(nn.ReLU()(self.normalization(self.fully_connected(x))))  # (batch, dim)
        return x

class Control_Group(nn.Module):
    def __init__(self, l_dim, v_dim, a_dim, dim, l_len, v_len, a_len, n_heads, n_layers, ffn):
        super().__init__()
        self.feature = Multi_class(l_dim=l_dim, v_dim=v_dim, a_dim=a_dim, dim=dim, l_len=l_len, v_len=v_len, a_len=a_len, n_heads=n_heads, n_layers=n_layers, ffn=ffn)
        self.emotion = nn.Linear(dim*6, 7)  # (no need sigmoid)
    def forward(self, l, v, a, l_mask, v_mask, a_mask):  # (batch, y_len, x_len, x_dim)
        out_list = []
        for i in range(l.shape[1]):
            temp_l, temp_v, temp_a, temp_l_mask, temp_v_mask, temp_a_mask = l[:,i,:,:], v[:,i,:,:], a[:,i,:,:], l_mask[:,i,:], v_mask[:,i,:], a_mask[:,i,:]
            temp_feat = self.feature(temp_l, temp_v, temp_a, temp_l_mask, temp_v_mask, temp_a_mask)  # (batch, dim)
            temp_emotion = self.emotion(temp_feat)  # (batch, 7)
            out_list.append(temp_emotion.unsqueeze(1))
        return torch.cat([torch.cat(out_list, dim=1), torch.cat(out_list, dim=1)], dim=-1)  # (batch, y_len, 7*2)

class State_Transfer(nn.Module):
    def __init__(self, l_dim, v_dim, a_dim, dim, l_len, v_len, a_len, n_heads, n_layers, ffn):
        super().__init__()
        self.feature = Multi_class(l_dim=l_dim, v_dim=v_dim, a_dim=a_dim, dim=dim, l_len=l_len, v_len=v_len, a_len=a_len, n_heads=n_heads, n_layers=n_layers, ffn=ffn)
        self.emotion = nn.Linear(dim*6, 7)  # (no need sigmoid)
        self.trans = nn.Parameter(torch.rand(7,7,7), requires_grad=True)
        self.out = nn.Linear(7, 7)  # (no need sigmoid)
    def forward(self, l, v, a, l_mask, v_mask, a_mask, p):  # (batch, y_len, x_len, x_dim)
        out_list, emo_list = [], []
        for i in range(l.shape[1]):
            temp_l, temp_v, temp_a, temp_l_mask, temp_v_mask, temp_a_mask = l[:,i,:,:], v[:,i,:,:], a[:,i,:,:], l_mask[:,i,:], v_mask[:,i,:], a_mask[:,i,:]
            temp_feat = self.feature(temp_l, temp_v, temp_a, temp_l_mask, temp_v_mask, temp_a_mask)  # (batch, dim)
            temp_emotion = self.emotion(temp_feat)  # (batch, 7)
            if i != 0:
                batch_list = []
                for j in range(temp_emotion.shape[0]):
                    if random.random() <= p:
                        temp_out = torch.matmul(emo_list[-1].squeeze(1)[j], nn.Tanh()(self.trans))  # (7, 7)
                    else:
                        temp_out = torch.matmul(out_list[-1].squeeze(1)[j], nn.Tanh()(self.trans))  # (7, 7)
                    temp_out = torch.matmul(nn.Softmax(dim=-1)(temp_emotion[j]), temp_out)  # (7,)
                    temp_out = self.out(temp_out)
                    batch_list.append(temp_out.unsqueeze(0))
                temp_energy = torch.cat(batch_list, dim=0)  # (batch, 7)
            else:
                temp_energy = temp_emotion
            out_list.append(temp_energy.unsqueeze(1))
            emo_list.append(temp_emotion.unsqueeze(1))
        return torch.cat([torch.cat(out_list, dim=1), torch.cat(emo_list, dim=1)], dim=-1)  # (batch, y_len, 7*2)

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

def train(model, iterator, optimizer, p):
    model.train()
    epoch_loss, count = 0, 0
    iter_bar = tqdm(iterator, desc='Training')
    for _, batch in enumerate(iter_bar):
        count += 1
        optimizer.zero_grad()
        linguistic, visual, acoustic, label, l_mask, v_mask, a_mask, mask = zip(*batch)
        linguistic, visual, acoustic, label, l_mask, v_mask, a_mask, mask = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic),\
        torch.cuda.LongTensor(label), torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(mask)
        logits_clsf = model(linguistic, visual, acoustic, l_mask, v_mask, a_mask, p)
        loss = multi_circle_loss(logits_clsf, label)
        loss = (loss*mask).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)  #梯度裁剪
        optimizer.step()
        iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
        epoch_loss += loss.item()
    return epoch_loss / count

def valid(model, iterator, p):
    model.eval()
    epoch_loss, count = 0, 0
    with torch.no_grad():
        iter_bar = tqdm(iterator, desc='Validation')
        for _, batch in enumerate(iter_bar):
            count += 1
            linguistic, visual, acoustic, label, l_mask, v_mask, a_mask, mask = zip(*batch)
            linguistic, visual, acoustic, label, l_mask, v_mask, a_mask, mask = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic),\
            torch.cuda.LongTensor(label), torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(mask)
            logits_clsf = model(linguistic, visual, acoustic, l_mask, v_mask, a_mask, p)
            loss = multi_circle_loss(logits_clsf[:, :, :7], label[:, :, :7])
            loss = (loss*mask).mean()
            iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss, count, epoch_loss / count

def run(model, data_set, train_list, valid_list, augment_list, batch_size, learning_rate, epochs, name):
    log_file = log_dir+name+'.txt'
    with open(log_file, 'w') as log_f:
        log_f.write('epoch, train_loss, valid_loss\n')
    writer = SummaryWriter(log_dir)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(epochs):
        print('Epoch: ' + str(epoch+1))
        train_iterator = data_loader(data_set, train_list, batch_size)
        valid_iterator = data_loader(data_set, valid_list, batch_size)
        p = Miu/(Miu-1+math.exp(epoch/3))  # epoch = miu + 1 时 p = 0.5
        train_loss = train(model, train_iterator, optimizer, p)
        _, _, valid_loss = valid(model, valid_iterator, 0)
        writer.add_scalars(name, {'train_loss':train_loss, 'valid_loss':valid_loss}, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss) 
        with open(log_file, 'a') as log_f:
            log_f.write('\n{epoch},{train_loss: 2.2f},{valid_loss: 2.2f}\n'.format(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss))
        if valid_loss == min(loss_list):
            stop = 0
            torch.save(model.state_dict(), os.path.join(log_dir, name+'_'+str(valid_loss)[:4]+'.pt'))
        else:
            stop += 1
            if stop >= 3:
                break
    writer.close()

# random.shuffle(train_name_list)
model_1 = State_Transfer(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
valid_list = train_name_list[:int(len(train_name_list)*0.2)]
train_list = train_name_list[int(len(train_name_list)*0.2):]
augment_list = augmentation(train_list)
run(model_1, data_set, train_list, valid_list, augment_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_1')
# model_2 = State_Transfer(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
# valid_list = train_name_list[int(len(train_name_list)*0.2):int(len(train_name_list)*0.4)]
# train_list = train_name_list[:int(len(train_name_list)*0.2)] + train_name_list[int(len(train_name_list)*0.4):]
# augment_list = augmentation(train_list)
# run(model_2, data_set, train_list, valid_list, augment_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_2')
# model_3 = State_Transfer(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
# valid_list = train_name_list[int(len(train_name_list)*0.4):int(len(train_name_list)*0.6)]
# train_list = train_name_list[:int(len(train_name_list)*0.4)] + train_name_list[int(len(train_name_list)*0.6):]
# augment_list = augmentation(train_list)
# run(model_3, data_set, train_list, valid_list, augment_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_3')
# model_4 = State_Transfer(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
# valid_list = train_name_list[int(len(train_name_list)*0.6):int(len(train_name_list)*0.8)]
# train_list = train_name_list[:int(len(train_name_list)*0.6)] + train_name_list[int(len(train_name_list)*0.8):]
# augment_list = augmentation(train_list)
# run(model_4, data_set, train_list, valid_list, augment_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_4')
# model_5 = State_Transfer(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
# valid_list = train_name_list[int(len(train_name_list)*0.8):]
# train_list = train_name_list[:int(len(train_name_list)*0.8)]
# augment_list = augmentation(train_list)
# run(model_5, data_set, train_list, valid_list, augment_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_5')

#  test
def test(model_1):
    model_1.eval()
    with torch.no_grad():
        for i in range(11):
            t = i/10-0.7
            test_iterator = data_loader(data_set, test_name_list, batch_size=BATCH)
            label_happ, soft_happ = [], []
            label_sadn, soft_sadn = [], []
            label_ange, soft_ange = [], []
            label_surp, soft_surp = [], []
            label_disg, soft_disg = [], []
            label_fear, soft_fear = [], []
            for _, batch in enumerate(test_iterator):
                linguistic, visual, acoustic, label, l_mask, v_mask, a_mask, mask = zip(*batch)
                linguistic, visual, acoustic, label, l_mask, v_mask, a_mask, mask = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic),\
                torch.cuda.LongTensor(label), torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(mask)
                pred = (model_1(linguistic, visual, acoustic, l_mask, v_mask, a_mask, 0)).detach().cpu()
                zero = torch.zeros_like(pred)
                one = torch.ones_like(pred)
                label = label.detach().cpu()
                happ_pred = torch.where(pred > t, one, zero)
                sadn_pred = torch.where(pred > t, one, zero)
                ange_pred = torch.where(pred > t, one, zero)
                surp_pred = torch.where(pred > t, one, zero)
                disg_pred = torch.where(pred > t, one, zero)
                fear_pred = torch.where(pred > t, one, zero)
                for j in range(mask.shape[0]):  # happ sadn ange disg surp fear neut
                    for k in range(mask.shape[1]):
                        if mask[j][k] == 1:
                            label_happ.append(int(label[j][k][0]))
                            label_sadn.append(int(label[j][k][1]))
                            label_ange.append(int(label[j][k][2]))
                            label_surp.append(int(label[j][k][4]))
                            label_disg.append(int(label[j][k][3]))
                            label_fear.append(int(label[j][k][5]))
                            soft_happ.append(int(happ_pred[j][k][0]))
                            soft_sadn.append(int(sadn_pred[j][k][1]))
                            soft_ange.append(int(ange_pred[j][k][2]))
                            soft_surp.append(int(surp_pred[j][k][4]))
                            soft_disg.append(int(disg_pred[j][k][3]))
                            soft_fear.append(int(fear_pred[j][k][5]))
            happ_f1 = f1_score(label_happ, soft_happ, average='weighted')
            sadn_f1 = f1_score(label_sadn, soft_sadn, average='weighted')
            ange_f1 = f1_score(label_ange, soft_ange, average='weighted')
            surp_f1 = f1_score(label_surp, soft_surp, average='weighted')
            disg_f1 = f1_score(label_disg, soft_disg, average='weighted')
            fear_f1 = f1_score(label_fear, soft_fear, average='weighted')
            print('t: ', t)
            print('happ_f1: ', happ_f1)
            print('sadn_f1: ', sadn_f1)
            print('ange_f1: ', ange_f1)
            print('fear_f1: ', fear_f1)
            print('disg_f1: ', disg_f1)
            print('surp_f1: ', surp_f1)
            print('\n')
    return 0

model_1 = State_Transfer(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_1.load_state_dict(torch.load(log_dir + '?.pt'))
test(model_1)
for name, p in model_1.named_parameters():
    if name == 'trans':
        print('Transfer matrix:', nn.Tanh()(p))
