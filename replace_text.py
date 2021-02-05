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

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
text_model = AutoModel.from_pretrained("bert-base-chinese")

# parameters
user = 'dango/multimodal'
mosei_dir = '/home/'+user+'/CMU-MOSEI/align/'
ren_dir = '/home/'+user+'/ren/1487_txt_hier_sents_202002/'
log_dir = '/home/'+user+'/CMU-MOSEI/par_log/simple_1/'
EPOCHS = 99
CLIP = 1.0
LR = 0.001
BATCH = 64
L_DIM = 768
V_DIM = 35
A_DIM = 74
L_LEN = 32
V_LEN = 32
A_LEN = 32
FFN = 2
DIM = 96
N_HEADS = 6
N_LAYERS = 1
DROP = 0.1

# data
mosei_dict = {'acoustic':mosei_dir+'COAVAREP.csd', 'visual':mosei_dir+'FACET 4.2.csd', 'label':mosei_dir+'All Labels.csd'}
mosei_set = mmdatasdk.mmdataset(mosei_dict)

train_name, test_name = [], []
for name in mosei_set.computational_sequences['label'].data.keys():
    if name.split('[')[0] in mmdatasdk.cmu_mosei.standard_folds.standard_test_fold:
        test_name.append(name)
    else:
        train_name.append(name)
train_name, test_name = list(set(train_name)), list(set(test_name))

def check_contain_chinese(strings):
    chinese = False
    for i in range(len(strings)):
        if u'\u4e00' <= strings[i] <= u'\u9fff':
            chinese = True
    return chinese

def ren_label_text(num):
    label_list, text_list = [], []
    with open(ren_dir + 'cet_' + str(num) + '.txt', 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            if line[0] == 's':
                if line.split(':')[2] == '\n' or line.split(':')[2] == '/n\n' or line.split(':')[2] == '/n' or line.split(':')[2] == '' or line.split(':')[2][0] == '/':
                    continue
                else:
                    temp_text = line.strip().split(':')[2].split('  ')
                    for i in range(len(temp_text)):
                        temp_text[i] = temp_text[i].split('/')[0]
                    if check_contain_chinese(temp_text) == False:
                        continue
                    else:
                        text_list.append(''.join(temp_text))
                        degree = line.split(':')[1].split(',')[:8]
                        temp_label = [0,0,0,0,0,0,0,0] # Love,Anxiety,Sorrow,Joy,Expect,Hate,Anger,Surprise
                        label = ['0', '0', '0', '0', '0', '0', '0']  # happ sadn ange disg surp fear neut
                        for index, x in enumerate(degree):
                            if x != '0.0':
                                temp_label[index] = 1
                        if sum(temp_label) == 0:
                            label[6] = '1'
                        if sum(temp_label) != 0 and int(temp_label[2]) == 1:
                            label[1] = '1'
                        if sum(temp_label) != 0 and int(temp_label[6]) == 1:
                            label[2] = '1'
                        if sum(temp_label) != 0 and int(temp_label[5]) == 1:
                            label[3] = '1'
                        if sum(temp_label) != 0 and int(temp_label[7]) == 1:
                            label[4] = '1'
                        if sum(temp_label) != 0 and int(temp_label[1]) == 1:
                            label[5] = '1'
                        if sum(temp_label) != 0 and (int(temp_label[0]) == 1 or int(temp_label[3]) == 1 or int(temp_label[4]) == 1):
                            label[0] = '1'
                        label_list.append(''.join(label))
    return label_list, text_list

def ren_text_to_dict():
    dicts = {}
    for i in tqdm(range(1, 1488)):
        if i == 490 or i == 761:
            continue
        label_list, text_list = ren_label_text(i)
        for j in range(len(label_list)):
            if label_list[j] not in dicts.keys():
                temp_list = [text_list[j]]
                dicts[label_list[j]] = temp_list
            else:
                temp_list = dicts[label_list[j]]
                temp_list.append(text_list[j])
                dicts[label_list[j]] = temp_list
    for key in dicts.keys():
        temp_list = dicts[key]
        random.shuffle(temp_list)
        dicts[key] = temp_list
    return dicts

ren_dict = ren_text_to_dict()

def label_processing(l):  # sent happ sadn ange surp disg fear
    label = ['0', '0', '0', '0', '0', '0', '0']  # happ sadn ange disg surp fear neut
    label[0] = '1' if l[1] > 0 else '0'
    label[1] = '1' if l[2] > 0 else '0'
    label[2] = '1' if l[3] > 0 else '0'
    label[3] = '1' if l[5] > 0 else '0'
    label[4] = '1' if l[4] > 0 else '0'
    label[5] = '1' if l[6] > 0 else '0'
    if sum(l) == 0.0:
        label[6] = '1'
    return label

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

def data_loader(data_set, replace_dict, name_list, batch_size):
    random.shuffle(name_list)
    count = 0
    while count < len(name_list):
        batch = []
        size = min(batch_size, len(name_list) - count)
        for _ in range(size):
            l_temp, v_temp, a_temp, lmask_temp, vmask_temp, amask_temp, label_temp = [], [], [], [], [], [], []
            label = label_processing(data_set.computational_sequences['label'].data[name_list[count]]["features"][0])
            if ''.join(label) != '0000001' and ''.join(label) in replace_dict.keys() and len(replace_dict[''.join(label)]) != 0:
                text = replace_dict[''.join(label)][-1]
                replace_dict[''.join(label)] = replace_dict[''.join(label)][:-1]
            else:
                text = replace_dict['0000001'][random.randint(0, len(replace_dict['0000001'])-1)]
            text = text_model(**text_tokenizer(text[:509], return_tensors="pt"))[0][0][1:-1,:].detach().numpy()
            l, l_mask = masking(text[-L_LEN:], L_LEN)
            v, v_mask = masking(data_set.computational_sequences['visual'].data[name_list[count]]["features"][-V_LEN:], V_LEN)
            a, a_mask = masking(data_set.computational_sequences['acoustic'].data[name_list[count]]["features"][-A_LEN:], A_LEN, is_audio=True)
            for i in range(len(label)):
                label[i] = int(label[i])
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
        self.classifier = nn.Linear(dim, 7)
    def forward(self, l, v, a, l_mask, v_mask, a_mask):
        l, v, a = self.unify_dimension(l, v, a)
        l = l + self.linguistic_position(l)
        v = v + self.visual_position(v)
        a = a + self.acoustic_position(a)
        ll, lv, la = l, l, l
        vv, vl, va = v, v, v
        aa, al, av = a, a, a
        l_list, v_list, a_list = [], [], []
        scores = None
        for i in range(self.n_layers):
            ll, scores = self.multimodal_blocks[self.n_layers*0+i](ll, l, l, l_mask, scores)
            l_list.append(ll)
        scores = None
        for i in range(self.n_layers):
            lv, scores = self.multimodal_blocks[self.n_layers*1+i](lv, v, v, v_mask, scores)
            l_list.append(lv)
        scores = None
        for i in range(self.n_layers):
            la, scores = self.multimodal_blocks[self.n_layers*2+i](la, a, a, a_mask, scores)
            l_list.append(la)
        scores = None
        for i in range(self.n_layers):
            vv, scores = self.multimodal_blocks[self.n_layers*3+i](vv, v, v, v_mask, scores)
            v_list.append(vv)
        scores = None
        for i in range(self.n_layers):
            vl, scores = self.multimodal_blocks[self.n_layers*4+i](vl, l, l, l_mask, scores)
            v_list.append(vl)
        scores = None
        for i in range(self.n_layers):
            va, scores = self.multimodal_blocks[self.n_layers*5+i](va, a, a, a_mask, scores)
            v_list.append(va)
        scores = None
        for i in range(self.n_layers):
            aa, scores = self.multimodal_blocks[self.n_layers*6+i](aa, a, a, a_mask, scores)
            a_list.append(aa)
        scores = None
        for i in range(self.n_layers):
            al, scores = self.multimodal_blocks[self.n_layers*7+i](al, l, l, l_mask, scores)
            a_list.append(al)
        scores = None
        for i in range(self.n_layers):
            av, scores = self.multimodal_blocks[self.n_layers*8+i](av, v, v, v_mask, scores)
            a_list.append(av)
        l = torch.cat(l_list, dim=2)  # (batch, l_len, dim*3*n_layers)
        v = torch.cat(v_list, dim=2)
        a = torch.cat(a_list, dim=2)
        x = torch.cat([l, a, v], dim=1)  # (batch, l_len+v_len+a_len, dim*3*n_layers)
        x = torch.cat([torch.mean(x, 1), torch.max(x, 1)[0]], dim=1)  # (batch, dim*6*n_layers)
        x = self.drop(nn.ReLU()(self.normalization(self.fully_connected(x))))  # (batch, dim)
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
        linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic),\
        torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(label)
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
            linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic),\
            torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(label)
            logits_clsf = model(linguistic, visual, acoustic, l_mask, v_mask, a_mask)
            loss = multi_circle_loss(logits_clsf, label)
            loss = loss.mean()
            iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss, count, epoch_loss / count

def run(model, data_set, replace_dict, train_list, valid_list, batch_size, learning_rate, epochs, name):
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
        train_iterator = data_loader(data_set, replace_dict, train_list, batch_size)
        valid_iterator = data_loader(data_set, replace_dict, valid_list, batch_size)
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
            if stop >= 3:
                break
    writer.close()

random.shuffle(train_name)
model_1 = Multi_class(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
run(model_1, mosei_set, ren_dict, train_name, test_name, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_1')
print('Finish')
