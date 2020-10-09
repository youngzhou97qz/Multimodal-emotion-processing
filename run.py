import os
import math
import random
import numpy as np
from tqdm import tqdm
from mmsdk import mmdatasdk

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.backends.cudnn.benchmark = True

# parameters
user = 'dango'
save_dir = '/home/'+user+'/multimodal/CMU-MOSEI/weight/'
L_DIM = 300
V_DIM = 35
A_DIM = 74
DIM = 192
L_LEN = 20
V_LEN = 200
A_LEN = 600
N_HEADS = 6
FFN = 2
N_LAYERS = 2
EPS = 1e-6
ACTIV = 'gelu'  # gelu & relu
UNIFY = 'Conv1D'  # Conv1D & FC
POS = 'True'  # True & False
POOL = 'avg_1'  # avg_1 & avg_2 & max_1 & max_2 & avg_1+max_1 & avg_2+max_2 & avg_1_cat_max_1 & avg_2_cat_max_2
BATCH = 64
LR = 0.001
CLIP = 1.0
EPOCHS = 9999

# data
data_dict={'linguistic':'/home/'+user+'/multimodal/CMU-MOSEI/align/glove_vectors.csd', 
       'acoustic':'/home/'+user+'/multimodal/CMU-MOSEI/align/COAVAREP.csd', 
       'visual':'/home/'+user+'/multimodal/CMU-MOSEI/align/FACET 4.2.csd', 
       'label':'/home/'+user+'/multimodal/CMU-MOSEI/align/All Labels.csd'}
data_set=mmdatasdk.mmdataset(data_dict)
train_name, valid_name, test_name = [], [], []
for name in data_set.computational_sequences['label'].data.keys():
    if name.split('[')[0] in mmdatasdk.cmu_mosei.standard_folds.standard_train_fold:
        train_name.append(name)
    elif name.split('[')[0] in mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold:
        valid_name.append(name)
    elif name.split('[')[0] in mmdatasdk.cmu_mosei.standard_folds.standard_test_fold:
        test_name.append(name)

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
            label = data_set.computational_sequences['label'].data[name_list[count]]["features"][0][1:]
            if len(l) >= l_len:
                l_mask = np.ones(l_len)
            else:
                l_mask = np.concatenate((np.zeros(l_len - len(l)), np.ones(len(l))))
            l = np.concatenate([np.zeros([l_len]+list(l.shape[1:])),l],axis=0)[-l_len:,...]
            if len(v) >= v_len:
                v_mask = np.ones(v_len)
            else:
                v_mask = np.concatenate((np.zeros(v_len - len(v)), np.ones(len(v))))
            v = np.concatenate([np.zeros([v_len]+list(v.shape[1:])),v],axis=0)[-v_len:,...]
            if len(a) >= a_len:
                a_mask = np.ones(a_len)
            else:
                a_mask = np.concatenate((np.zeros(a_len - len(a)), np.ones(len(a))))
            a = np.concatenate([np.zeros([a_len]+list(a.shape[1:])),a],axis=0)[-a_len:,...]
            for i in range(len(a)):
                for j in range(len(a[i])):
                    if math.isinf(a[i][j]):
                        a[i][j] = -70.
            batch.append((l, v, a, l_mask, v_mask, a_mask, label))
            count += 1
        yield batch

# model 1
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class Unify_Dimension_Conv1d(nn.Module):
    def __init__(self, l_dim, v_dim, a_dim, dim=768):
        super().__init__()
        self.linguistic = nn.Conv1d(l_dim, dim, kernel_size=1, bias=False)
        self.visual = nn.Conv1d(v_dim, dim, kernel_size=1, bias=False)
        self.acoustic = nn.Conv1d(a_dim, dim, kernel_size=1, bias=False)
    def forward(self, l, v, a):
        l, v, a = l.transpose(1, 2), v.transpose(1, 2), a.transpose(1, 2)
        l, v, a = self.linguistic(l), self.visual(v), self.acoustic(a)
        return l.transpose(1, 2), v.transpose(1, 2), a.transpose(1, 2)

class Unify_Dimension_FC(nn.Module):
    def __init__(self, l_dim, v_dim, a_dim, dim=768):
        super().__init__()
        self.linguistic = nn.Linear(l_dim, dim)
        self.visual = nn.Linear(v_dim, dim)
        self.acoustic = nn.Linear(a_dim, dim)
    def forward(self, l, v, a):
        return self.linguistic(l), self.visual(v), self.acoustic(a)

class Position_Embedding(nn.Module):
    def __init__(self, max_len, dim=768):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, dim)
        self.len = max_len
    def forward(self, x):
        position_ids = torch.arange(self.len, device=device).unsqueeze(0).repeat(x.size()[0],1)
        return self.position_embeddings(position_ids.to(device))

class Modal_Embedding(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.type_embeddings = nn.Embedding(3, dim)
    def forward(self, type_ids):
        return self.type_embeddings(type_ids)

class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, dim=768, n_heads=12):
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
    def __init__(self, dim=768, ffn=4, activation='gelu'):
        super().__init__()
        self.fully_connected_1 = nn.Linear(dim, dim*ffn)
        self.fully_connected_2 = nn.Linear(dim*ffn, dim)
        self.activation = activation
    def forward(self, x):
        if self.activation == 'gelu':
            return self.fully_connected_2(self.gelu(self.fully_connected_1(x)))
        elif self.activation == 'relu':
            return self.fully_connected_2(nn.ReLU()(self.fully_connected_1(x)))
    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Transformer_Blocks(nn.Module):
    def __init__(self, dim=768, eps=1e-6, n_heads=12, n_layers=12, ffn=4, activation='gelu', mode='not_cross'):
        super().__init__()
        self.normalization = nn.ModuleList([nn.LayerNorm(dim, eps=eps) for _ in range(n_layers*4)])
        self.self_attention = nn.ModuleList([Multi_Head_Self_Attention(dim=dim, n_heads=n_heads) for _ in range(n_layers)])
        self.fully_connected = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.feed_forward = nn.ModuleList([Position_Wise_Feed_Forward(dim=dim, ffn=ffn, activation='gelu') for _ in range(n_layers)])
        self.mode = mode
    def forward(self, q, k, v, mask, layer_num):
        if self.mode == 'cross':
            q, k, v = self.normalization[layer_num*4+0](q), self.normalization[layer_num*4+1](k), self.normalization[layer_num*4+2](v)
        elif self.mode == 'not_cross':
            q, k, v = self.normalization[layer_num*4+0](q), self.normalization[layer_num*4+1](q), self.normalization[layer_num*4+2](q)
        q = q + self.fully_connected[layer_num](self.self_attention[layer_num](q, k, v, mask))
        q = q + self.feed_forward[layer_num](self.normalization[layer_num*4+3](q))
        return q

class Model_1(nn.Module):
    def __init__(self, l_dim, v_dim, a_dim, dim, l_len, v_len, a_len, eps, n_heads, n_layers, ffn, unify_dimension='Conv1D', position='True', activation='gelu', pooling='avg_1'):
        super().__init__()
        if unify_dimension == 'Conv1D':
            self.unify_dimension = Unify_Dimension_Conv1d(l_dim, v_dim, a_dim, dim)
        elif unify_dimension == 'FC':
            self.unify_dimension = Unify_Dimension_FC(l_dim, v_dim, a_dim, dim)
        self.position = position
        if position:
            self.linguistic_position = Position_Embedding(l_len, dim)
            self.visual_position = Position_Embedding(v_len, dim)
            self.acoustic_position = Position_Embedding(a_len, dim)
        self.n_layers = n_layers
        self.transformer_blocks = nn.ModuleList([Transformer_Blocks(dim, eps, n_heads, n_layers, ffn, activation, mode='not_cross') for _ in range(3)])
        self.pooling = pooling
        if pooling == 'avg_1' or 'max_1' or 'avg_1+max_1':
            self.fully_connected = nn.Linear(dim, dim)
        elif pooling == 'avg_2' or 'max_2' or 'avg_2+max_2':
            self.fully_connected = nn.Linear(l_len+v_len+a_len, dim)
        elif pooling == 'avg_1_cat_max_1':
            self.fully_connected = nn.Linear(dim*2, dim)
        elif pooling == 'avg_2_cat_max_2':
            self.fully_connected = nn.Linear((l_len+v_len+a_len)*2, dim)
        self.normalization = nn.LayerNorm(dim, eps=eps)
        self.classifier = nn.Linear(dim, 6)
    def forward(self, l, v, a, l_mask, v_mask, a_mask):
        l, v, a = self.unify_dimension(l, v, a)
        if self.position:
            l = l + self.linguistic_position(l)
            v = v + self.visual_position(v)
            a = a + self.acoustic_position(a)
        for i in range(self.n_layers):
            l = self.transformer_blocks[0](l, l, l, l_mask, i)
            v = self.transformer_blocks[0](v, v, v, v_mask, i)
            a = self.transformer_blocks[0](a, a, a, a_mask, i)
        x = torch.cat([l, a, v], dim=1)
        if self.pooling=='avg_1':
            x = torch.mean(x, 1)
        elif self.pooling=='avg_2':
            x = torch.mean(x, 2)
        elif self.pooling=='max_1':
            x = torch.max(x, 1)[0]
        elif self.pooling=='max_2':
            x = torch.max(x, 2)[0]
        elif self.pooling=='avg_1+max_1':
            x = torch.mean(x, 1) + torch.max(x, 1)[0]
        elif self.pooling=='avg_2+max_2':
            x = torch.mean(x, 2) + torch.max(x, 2)[0]
        elif self.pooling=='avg_1_cat_max_1':
            x = torch.cat([torch.mean(x, 1), torch.max(x, 1)[0]], dim=1)
        elif self.pooling=='avg_2_cat_max_2':
            x = torch.cat([torch.mean(x, 2), torch.max(x, 2)[0]], dim=1)
        x = nn.ReLU()(self.normalization(self.fully_connected(x)))
        return self.classifier(x)

# run
def train(model, iterator, optimizer):
    model.train()
    epoch_loss, count = 0, 0
    iter_bar = tqdm(iterator, desc='Training')
    for _, batch in enumerate(iter_bar):
        count += 1
        optimizer.zero_grad()
        linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = zip(*batch)
        linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.FloatTensor(label)
        logits_clsf = model(linguistic, visual, acoustic, l_mask, v_mask, a_mask)
#         loss = nn.BCELoss()(nn.Sigmoid()(logits_clsf), label)
        loss = nn.L1Loss()(logits_clsf, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)  #梯度裁剪
#         nn.utils.clip_grad_value_(model.parameters(), CLIP)
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
            linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.FloatTensor(label)
            logits_clsf = model(linguistic, visual, acoustic, l_mask, v_mask, a_mask)
            loss = nn.L1Loss()(logits_clsf, label)
            iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss / count

def run(model, data_set, train_list, valid_list, batch_size, learning_rate, epochs):
    log_name = 'gelu_conv_avg_pos_norm_bce_loss_'
    writer = SummaryWriter('/home/'+user+'/multimodal/CMU-MOSEI/log/')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(epochs):
        train_iterator = data_loader(data_set, train_list, batch_size, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN)
        valid_iterator = data_loader(data_set, valid_list, batch_size, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN)
        train_loss = train(model, train_iterator, optimizer)
        valid_loss = valid(model, valid_iterator)
        writer.add_scalars(log_name, {'train_loss':train_loss, 'valid_loss':valid_loss}, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss) 
        if valid_loss == min(loss_list):
            stop = 0
            torch.save(model.state_dict(), os.path.join(save_dir, log_name+str(valid_loss)[:4]+'.pt'))
            print('Epoch: ' + str(epoch+1) + ', Loss: ' + str(valid_loss)[:4])
        else:
            stop += 1
            if stop >= 5:
                break
    writer.close()
    return min(loss_list)

model = Model_1(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN, unify_dimension=UNIFY, position=POS, activation=ACTIV, pooling=POOL).to(device)
print(get_parameter_number(model))
print('Training set: ', len(train_name))
print('Validation set: ', len(valid_name))
cls_loss = run(model, data_set, train_name, valid_name, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS)
#  tensorboard --logdir=/home/dango/multimodal/CMU-MOSEI/log --port 8123
