import os
import math
import random
import pickle
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

# 目录
user = 'XX/multimodal/CMU-MOSEI'
data_dir = '/home/'+user+'/align/'
log_dir = '/home/'+user+'/log/XX/'
label_file = data_dir+'labels.txt'

# 参数
EPOCHS = 999
CLIP = 1.0
LR = 0.001
L_LEN = 20
V_LEN = 100
A_LEN = 200
L_DIM = 300
V_DIM = 35
A_DIM = 74
DIM = 96
BATCH = 64
DROP = 0.0
FFN = 1
N_HEADS = 6
N_LAYERS = 1

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

# 句子排序，分出训练测试集18586/4662，给出标签 happy, sad, angry, disgust, surprise, fear, neutral
def name_list_and_label(label_file, test_name):
    train_name_list, test_name_list = [], []
    label_dict = {}
    with open(label_file, 'r') as f:
        lines = f.readlines()[1:]
        last_para = ''
        sent_list, time_list = [], []
        for line in tqdm(lines):
            sentence = line.split(',')[0]
            paragraph = sentence.split('[')[0]
            if paragraph == last_para:
                sent_list.append(sentence)
                time_list.append(float(line.split(',')[1]))
            else:
                if len(sent_list) > 0:
                    order_sent_list = [x for _,x in sorted(zip(time_list, sent_list))]
                    order_sent_list.insert(0, 'no_name')
                    for i in range(len(order_sent_list)-1):
                        temp_list = []
                        temp_list.append(order_sent_list[i])
                        temp_list.append(order_sent_list[i+1])
                        if last_para in test_name:
                            test_name_list.append(temp_list)
                        else:
                            train_name_list.append(temp_list)
                    sent_list, time_list = [], []
                sent_list.append(sentence)
                time_list.append(float(line.split(',')[1]))
            label_list = line.strip().split(',')[3:]
            for j in range(len(label_list)):
                label_list[j] = int(label_list[j])
            label_dict[sentence] = label_list
            last_para = paragraph
        order_sent_list = [x for _,x in sorted(zip(time_list, sent_list))]
        order_sent_list.insert(0, 'no_name')
        for i in range(len(order_sent_list)-1):
            temp_list = []
            temp_list.append(order_sent_list[i])
            temp_list.append(order_sent_list[i+1])
            if last_para in test_name:
                test_name_list.append(temp_list)
            else:
                train_name_list.append(temp_list)
    return train_name_list, test_name_list, label_dict

train_name_list, test_name_list, label_dict = name_list_and_label(label_file, test_name)

def masking(m, m_len, is_bert=False, is_audio=False):
    feat, feat_mask = [], []
    if is_audio:
        for i in range(len(m)):
            for j in range(len(m[i])):
                if math.isinf(m[i][j]) or math.isnan(m[i][j]):
                    m[i][j] = -71.
    if is_bert:
        m_max = m[1:-1].max(axis=0)
        m_min = m[1:-1].min(axis=0)
        m_mean = m[1:-1].mean(axis=0)
        if len(m) > m_len-5:
            m_mask = np.ones(m_len)
            m_0 = m[1:m_len-4]
            m_0 = np.concatenate((np.expand_dims(m_max, axis=0), np.expand_dims(m_min, axis=0), np.expand_dims(m_mean, axis=0), np.expand_dims(m[0], axis=0), m_0, np.expand_dims(m[-1], axis=0)), axis=0)
            feat.append(m_0)
            feat_mask.append(m_mask)
            m_1 = m[len(m)-m_len+4:-1]
            m_1 = np.concatenate((np.expand_dims(m_max, axis=0), np.expand_dims(m_min, axis=0), np.expand_dims(m_mean, axis=0), np.expand_dims(m[0], axis=0), m_1, np.expand_dims(m[-1], axis=0)), axis=0)
            feat.append(m_1)
            feat_mask.append(m_mask)
        else:
            m_mask = np.concatenate((np.ones(len(m)+3), np.zeros(m_len - len(m)-3)))
            m = np.concatenate((np.expand_dims(m_max, axis=0), np.expand_dims(m_min, axis=0), np.expand_dims(m_mean, axis=0), m), axis=0)
            m = np.concatenate([m, np.zeros([m_len]+list(m.shape[1:]))],axis=0)[:m_len,...]
            feat.append(m)
            feat_mask.append(m_mask)
    else:
        m_max = m.max(axis=0)
        m_min = m.min(axis=0)
        m_mean = m.mean(axis=0)
        if len(m) >= m_len-3:
            m_mask = np.ones(m_len)
            m_0 = m[:m_len-3]
            m_0 = np.concatenate((np.expand_dims(m_max, axis=0), np.expand_dims(m_min, axis=0), np.expand_dims(m_mean, axis=0), m_0), axis=0)
            feat.append(m_0)
            feat_mask.append(m_mask)
            m_1 = m[len(m)-m_len+3:]
            m_1 = np.concatenate((np.expand_dims(m_max, axis=0), np.expand_dims(m_min, axis=0), np.expand_dims(m_mean, axis=0), m_1), axis=0)
            feat.append(m_1)
            feat_mask.append(m_mask)
        else:
            m_mask = np.concatenate((np.ones(len(m)+3), np.zeros(m_len - len(m)-3)))
            m = np.concatenate((np.expand_dims(m_max, axis=0), np.expand_dims(m_min, axis=0), np.expand_dims(m_mean, axis=0), m), axis=0)
            m = np.concatenate([m, np.zeros([m_len]+list(m.shape[1:]))],axis=0)[:m_len,...]
            feat.append(m)
            feat_mask.append(m_mask)
    return feat, feat_mask

# 数据加载器
def data_loader(name_list, label_dict, batch_size):
    random.shuffle(name_list)   # 随机注释
    count = 0
    while count < len(name_list):
        batch = []
        size = min(batch_size, len(name_list) - count)
        for _ in range(size):
            label = label_dict[name_list[count][1]]
            if name_list[count][0] == 'no_name':
                l_0 = [np.zeros((L_LEN, data_set.computational_sequences['linguistic'].data[name_list[count][1]]["features"][:].shape[1]))]
                l_0_mask = [np.zeros(L_LEN)]
                v_0 = [np.zeros((V_LEN, data_set.computational_sequences['visual'].data[name_list[count][1]]["features"][:].shape[1]))]
                v_0_mask = [np.zeros(V_LEN)]
                a_0 = [np.zeros((A_LEN, data_set.computational_sequences['acoustic'].data[name_list[count][1]]["features"][:].shape[1]))]
                a_0_mask = [np.zeros(A_LEN)]
            else:
                l_0 = data_set.computational_sequences['linguistic'].data[name_list[count][0]]["features"][:]
                l_0, l_0_mask = masking(l_0, L_LEN, is_bert=False, is_audio=False)
                v_0 = data_set.computational_sequences['visual'].data[name_list[count][0]]["features"][:]
                v_0, v_0_mask = masking(v_0, V_LEN, is_bert=False, is_audio=False)
                a_0 = data_set.computational_sequences['acoustic'].data[name_list[count][0]]["features"][:]
                a_0, a_0_mask = masking(a_0, A_LEN, is_bert=False, is_audio=True)
            l_1 = data_set.computational_sequences['linguistic'].data[name_list[count][1]]["features"][:]
            l_1, l_1_mask = masking(l_1, L_LEN, is_bert=False, is_audio=False)
            v_1 = data_set.computational_sequences['visual'].data[name_list[count][1]]["features"][:]
            v_1, v_1_mask = masking(v_1, V_LEN, is_bert=False, is_audio=False)
            a_1 = data_set.computational_sequences['acoustic'].data[name_list[count][1]]["features"][:]
            a_1, a_1_mask = masking(a_1, A_LEN, is_bert=False, is_audio=True)
            if len(l_1_mask) > 1:
                l = np.concatenate((np.expand_dims(l_0[-1], axis=0), np.expand_dims(l_1[-1], axis=0)), axis=0)
                l_mask = np.concatenate((np.expand_dims(l_0_mask[-1], axis=0), np.expand_dims(l_1_mask[-1], axis=0)), axis=0)
                v = np.concatenate((np.expand_dims(v_0[-1], axis=0), np.expand_dims(v_1[-1], axis=0)), axis=0)
                v_mask = np.concatenate((np.expand_dims(v_0_mask[-1], axis=0), np.expand_dims(v_1_mask[-1], axis=0)), axis=0)
                a = np.concatenate((np.expand_dims(a_0[-1], axis=0), np.expand_dims(a_1[-1], axis=0)), axis=0)
                a_mask = np.concatenate((np.expand_dims(a_0_mask[-1], axis=0), np.expand_dims(a_1_mask[-1], axis=0)), axis=0)
                batch.append((l, v, a, l_mask, v_mask, a_mask, label))
            l = np.concatenate((np.expand_dims(l_0[0], axis=0), np.expand_dims(l_1[0], axis=0)), axis=0)
            l_mask = np.concatenate((np.expand_dims(l_0_mask[0], axis=0), np.expand_dims(l_1_mask[0], axis=0)), axis=0)
            v = np.concatenate((np.expand_dims(v_0[0], axis=0), np.expand_dims(v_1[0], axis=0)), axis=0)
            v_mask = np.concatenate((np.expand_dims(v_0_mask[0], axis=0), np.expand_dims(v_1_mask[0], axis=0)), axis=0)
            a = np.concatenate((np.expand_dims(a_0[0], axis=0), np.expand_dims(a_1[0], axis=0)), axis=0)
            a_mask = np.concatenate((np.expand_dims(a_0_mask[0], axis=0), np.expand_dims(a_1_mask[0], axis=0)), axis=0)
            batch.append((l, v, a, l_mask, v_mask, a_mask, label))
            count += 1
        yield batch

# 计算参数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# 统一 hidden size
class Unify_Dimension(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linguistic = nn.Linear(L_DIM, dim, bias=False)
        self.visual = nn.Linear(V_DIM, dim, bias=False)
        self.acoustic = nn.Linear(A_DIM, dim, bias=False)
    def forward(self, l, v, a):
        return self.linguistic(l), self.visual(v), self.acoustic(a)

# 注意力
class Attention_Block(nn.Module):
    def __init__(self, dim, n_heads, ffn):
        super().__init__()
        self.n_heads = n_heads
        self.drop = nn.Dropout(DROP)
        self.proj = nn.Linear(dim, dim, bias = False)
        self.minus = nn.Linear(dim*2, dim, bias = False)
        self.norm1 = nn.LayerNorm(dim)
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
        q = torch.cat([q, x], dim=-1)
        q = self.drop(self.norm1(self.minus(q)))
        return q, scores

# 多分类模型
class Multi_ATTN(nn.Module):
    def __init__(self, dim, l_len, v_len, a_len, n_heads, n_layers, ffn):
        super().__init__()
        self.unify_dimension = Unify_Dimension(dim)
        self.n_layers = n_layers
        self.multimodal_blocks = nn.ModuleList([Attention_Block(dim, n_heads, ffn) for _ in range(9*n_layers)])
        self.classifier = nn.Linear(dim*6*n_layers, 7, bias=False)  # dim*6*n_layers  dim
    def forward(self, l, v, a, l_mask, v_mask, a_mask):
        l, v, a = self.unify_dimension(l, v, a)
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
        return self.classifier(x)

class Concat_Trans(nn.Module):
    def __init__(self, dim, l_len, v_len, a_len, n_heads, n_layers, ffn):
        super().__init__()
        self.intensity = Multi_ATTN(dim, l_len, v_len, a_len, n_heads, n_layers, ffn)
        self.stimulation = Multi_ATTN(dim, l_len, v_len, a_len, n_heads, n_layers, ffn)
        self.trans = nn.Parameter(torch.rand(7,7,7), requires_grad=True)
        self.norm1 = nn.LayerNorm(7)
        self.out = nn.Linear(14, 7)
    def forward(self, l, v, a, l_mask, v_mask, a_mask):  # (batch, len, dim)
        last_feat = self.intensity(l[:,0,:,:], v[:,0,:,:], a[:,0,:,:], l_mask[:,0], v_mask[:,0], a_mask[:,0])
        this_feat = self.stimulation(l[:,1,:,:], v[:,1,:,:], a[:,1,:,:], l_mask[:,1], v_mask[:,1], a_mask[:,1])  # (batch, 7)
        batch_list = []
        for i in range(this_feat.shape[0]):
            temp_feat = torch.matmul(last_feat[i], self.trans)  # (7, 7)
            temp_feat = torch.matmul(this_feat[i], temp_feat)  # (7,)
            batch_list.append(temp_feat.unsqueeze(0))
        out_feat = torch.cat(batch_list, dim=0)  # (batch, 7)
        out_feat = torch.cat([this_feat, self.norm1(out_feat)], dim=1)
        return self.out(out_feat)

# 损失函数
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

# 训练
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

# 验证
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

# 训练器
def run(model, train_list, valid_list, label_dict, batch_size, learning_rate, epochs, log_name):
    log_file = log_dir+log_name+'.txt'
    with open(log_file, 'w') as log_f:
        log_f.write('epoch, train_loss, valid_loss\n')
    writer = SummaryWriter(log_dir)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(epochs):
        print('Epoch: ' + str(epoch+1))
        train_iterator = data_loader(train_list, label_dict, batch_size)
        valid_iterator = data_loader(valid_list, label_dict, batch_size)
        train_loss = train(model, train_iterator, optimizer)
        _, _, valid_loss = valid(model, valid_iterator)
        writer.add_scalars(log_name, {'train_loss':train_loss, 'valid_loss':valid_loss}, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss) 
        with open(log_file, 'a') as log_f:
            log_f.write('\n{epoch},{train_loss: 2.2f},{valid_loss: 2.2f}\n'.format(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss))
        if valid_loss == min(loss_list) and valid_loss > 0.009:
            stop = 0
            torch.save(model.state_dict(), os.path.join(log_dir, log_name+'_'+str(valid_loss)[:4]+'.pt'))
        else:
            stop += 1
            if stop >= 9:
                break
    writer.close()

# 训练模型
random.shuffle(train_name_list)  # 随机注释

model_1 = Concat_Trans(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
valid_list = train_name_list[:4096]
train_list = train_name_list[4096:]
run(model_1, train_list, valid_list, label_dict, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, log_name='model_1')

model_2 = Concat_Trans(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
valid_list = train_name_list[4096:8192]
train_list = train_name_list[:4096] + train_name_list[8192:]
run(model_2, train_list, valid_list, label_dict, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, log_name='model_2')

model_3 = Concat_Trans(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
valid_list = train_name_list[8192:12288]
train_list = train_name_list[:8192] + train_name_list[12288:]
run(model_3, train_list, valid_list, label_dict, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, log_name='model_3')

model_4 = Concat_Trans(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
valid_list = train_name_list[12288:16384]
train_list = train_name_list[:12288] + train_name_list[16384:]
run(model_4, train_list, valid_list, label_dict, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, log_name='model_4')
print('Finish')

model_1 = Concat_Trans(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_1.load_state_dict(torch.load(log_dir + 'model_1_2.18.pt'))
model_2 = Concat_Trans(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_2.load_state_dict(torch.load(log_dir + 'model_2_2.20.pt'))
model_3 = Concat_Trans(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_3.load_state_dict(torch.load(log_dir + 'model_3_2.22.pt'))
model_4 = Concat_Trans(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_4.load_state_dict(torch.load(log_dir + 'model_4_2.17.pt'))

# 测试阈值t
def test(model_1, model_2, model_3, model_4):
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_name_list, label_dict, batch_size=1)
        label_happ, soft_happ = [], []
        label_sadn, soft_sadn = [], []
        label_ange, soft_ange = [], []
        label_surp, soft_surp = [], []
        label_disg, soft_disg = [], []
        label_fear, soft_fear = [], []
        for _, batch in tqdm(enumerate(test_iterator)):
            linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = zip(*batch)
            linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic),\
            torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(label)
            pred_1 = (model_1(linguistic, visual, acoustic, l_mask, v_mask, a_mask)).detach().cpu()
            pred_2 = (model_2(linguistic, visual, acoustic, l_mask, v_mask, a_mask)).detach().cpu()
            pred_3 = (model_3(linguistic, visual, acoustic, l_mask, v_mask, a_mask)).detach().cpu()
            pred_4 = (model_4(linguistic, visual, acoustic, l_mask, v_mask, a_mask)).detach().cpu()
            pred = torch.mean(((pred_1+pred_2+pred_3+pred_4)/4), 0)
            zero = torch.zeros_like(pred)
            one = torch.ones_like(pred)
            label = label.detach().cpu()[0]
            pred_happ = torch.where(pred > (0.1), one, zero)
            pred_sadn = torch.where(pred > (-0.3), one, zero)
            pred_ange = torch.where(pred > (-0.5), one, zero)
            pred_surp = torch.where(pred > (-0.6), one, zero)
            pred_disg = torch.where(pred > (-0.3), one, zero)
            pred_fear = torch.where(pred > (-0.5), one, zero)
            label_happ.append(int(label[0]))
            label_sadn.append(int(label[1]))
            label_ange.append(int(label[2]))
            label_surp.append(int(label[4]))
            label_disg.append(int(label[3]))
            label_fear.append(int(label[5]))
            soft_happ.append(int(pred_happ[0]))
            soft_sadn.append(int(pred_sadn[1]))
            soft_ange.append(int(pred_ange[2]))
            soft_surp.append(int(pred_surp[4]))
            soft_disg.append(int(pred_disg[3]))
            soft_fear.append(int(pred_fear[5]))
    print('happ_acc: ', accuracy_score(label_happ, soft_happ))
    print('happ_f1: ', f1_score(label_happ, soft_happ, average='weighted'))
    print('sadn_acc: ', accuracy_score(label_sadn, soft_sadn))
    print('sadn_f1: ', f1_score(label_sadn, soft_sadn, average='weighted'))
    print('ange_acc: ', accuracy_score(label_ange, soft_ange))
    print('ange_f1: ', f1_score(label_ange, soft_ange, average='weighted'))
    print('fear_acc: ', accuracy_score(label_fear, soft_fear))
    print('fear_f1: ', f1_score(label_fear, soft_fear, average='weighted'))
    print('disg_acc: ', accuracy_score(label_disg, soft_disg))
    print('disg_f1: ', f1_score(label_disg, soft_disg, average='weighted'))
    print('surp_acc: ', accuracy_score(label_surp, soft_surp))
    print('surp_f1: ', f1_score(label_surp, soft_surp, average='weighted'))
    return 0

test(model_1, model_2, model_3, model_4)
