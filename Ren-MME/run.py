# import
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

file_path = '/home/dango/multimodal/Semi-automatic_labeling/wmyedjl/'
label_file = file_path + 'data/zero_one_adjust.csv'
text_path = file_path + 'text_feat/'
video_path = file_path + 'video_feat/'
audio_path = file_path + 'audio_feat/'
log_path = file_path + 'shixu_base/'

EPOCHS = 999
CLIP = 1.0
LR = 1e-3
L_LEN = 40
V_LEN = 76
A_LEN = 275
L_DIM = 768
V_DIM = 640
A_DIM = 205
DIM = 128
BATCH = 16
DROP = 0.1
FFN = 1
N_HEADS = 8
N_LAYERS = 1

# label
label_df = pd.read_csv(label_file)
train_set, test_set = [], []
for i in range(3870):
    name = str(label_df['Episode'][i]) + '_' + str(label_df['Dialogue'][i]) + '_' + str(label_df['Sentence'][i])
    label = [int(label_df['Love'][i]), int(label_df['Anxiety'][i]), int(label_df['Sorrow'][i]), int(label_df['Joy'][i]), int(label_df['Expect'][i]),
             int(label_df['Hate'][i]), int(label_df['Anger'][i]), int(label_df['Surprise'][i]), int(label_df['Neutral'][i])]
    if int(label_df['Episode'][i]) == 9 or int(label_df['Episode'][i]) == 10:
        test_set.append([name, label])
    else:
        train_set.append([name, label])

# text max 50
def text_features(name, text_len=L_LEN):
    # 固定长度
    temp_list = []
    temp_feat = np.load(text_path + name + '.npy')
    if len(temp_feat) < text_len:
        temp_list.append(temp_feat)
        pad_len = text_len - len(temp_feat)
        temp_list.append(np.zeros((pad_len, L_DIM)))
        feat = np.concatenate(temp_list, axis=0)
        mask = np.concatenate((np.ones(len(temp_feat)), np.zeros(pad_len)), axis=0)
    else:
        feat = temp_feat[:text_len]
        mask = np.ones(text_len)
    return feat, mask

# video max 77
def video_features(name, video_len=V_LEN):
    # 无特征时，取前后特征
    name_list = name.split('_')
    name_list[-1] = str(int(name_list[-1])-1)
    pre_name = '_'.join(name_list)
    name_list[-1] = str(int(name_list[-1])+2)
    pro_name = '_'.join(name_list)
    name_list[-1] = str(int(name_list[-1])-3)
    pre_pre_name = '_'.join(name_list)
    if os.path.exists(video_path + name + '.npy') == False:
        if os.path.exists(video_path + pre_name + '.npy') == False:
            if os.path.exists(video_path + pro_name + '.npy') == False:
                if os.path.exists(video_path + pre_pre_name + '.npy') == False:
                    temp_feat = np.zeros((video_len, V_DIM))
                else:
                    temp_feat = np.load(video_path + pre_pre_name + '.npy')
            else:
                temp_feat = np.load(video_path + pro_name + '.npy')
        else:
            temp_feat = np.load(video_path + pre_name + '.npy')
    else:
        temp_feat = np.load(video_path + name + '.npy')
    # 固定长度
    temp_list = []
    if len(temp_feat) < video_len:
        temp_list.append(temp_feat)
        pad_len = video_len - len(temp_feat)
        temp_list.append(np.zeros((pad_len, V_DIM)))
        feat = np.concatenate(temp_list, axis=0)
        mask = np.concatenate((np.ones(len(temp_feat)), np.zeros(pad_len)), axis=0)
    else:
        feat = temp_feat[:video_len]
        mask = np.ones(video_len)
    return feat, mask

# audio max 285
def audio_features(name, audio_len=A_LEN):
    # 固定长度
    temp_list = []
    temp_feat = np.load(audio_path + name + '.npy')
    temp_feat = np.transpose(temp_feat)
    if len(temp_feat) < audio_len:
        temp_list.append(temp_feat)
        pad_len = audio_len - len(temp_feat)
        temp_list.append(np.zeros((pad_len, A_DIM)))
        feat = np.concatenate(temp_list, axis=0)
        mask = np.concatenate((np.ones(len(temp_feat)), np.zeros(pad_len)), axis=0)
    else:
        feat = temp_feat[:audio_len]
        mask = np.ones(audio_len)
    return feat, mask

# loader
def data_loader(data_set, batch_size):
    random.shuffle(data_set)
    count = 0
    while count < len(data_set):
        batch = []
        size = min(batch_size, len(data_set) - count)
        for _ in range(size):
            pro_name = data_set[count][0]
            name_list = pro_name.split('_')
            if name_list[-1] == '1':
                pre_name = pro_name
            else:
                name_list[-1] = str(int(name_list[-1])-1)
                pre_name = '_'.join(name_list)
            pre_text_feat, pre_text_mask = text_features(pre_name)
            pro_text_feat, pro_text_mask = text_features(pro_name)
            pre_video_feat, pre_video_mask = video_features(pre_name)
            pro_video_feat, pro_video_mask = video_features(pro_name)
            pre_audio_feat, pre_audio_mask = audio_features(pre_name)
            pro_audio_feat, pro_audio_mask = audio_features(pro_name)
            batch.append((pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
                          pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, data_set[count][1]))
            batch.append((pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
                          pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, data_set[count][1]))
            count += 1
        yield batch
        
# model
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
        self.norm1 = nn.LayerNorm(dim)
    def forward(self, l, v, a):
        return self.norm1(self.linguistic(l)), self.norm1(self.visual(v)), self.norm1(self.acoustic(a))

# 注意力模块
class Attention_Block(nn.Module):
    def __init__(self, dim, n_heads, ffn):
        super().__init__()
        self.n_heads = n_heads
        self.drop = nn.Dropout(DROP)
        self.proj = nn.Linear(dim, dim, bias = False)
        self.minus = nn.Linear(dim*2, dim, bias = False)
        self.norm2 = nn.LayerNorm(dim)
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
        q = self.drop(self.norm2(self.minus(q)))
        return q, scores

# 交叉模型
class Multi_ATTN(nn.Module):
    def __init__(self, dim, l_len, v_len, a_len, n_heads, n_layers, ffn):
        super().__init__()
        self.unify_dimension = Unify_Dimension(dim)
        self.n_layers = n_layers
        self.multimodal_blocks = nn.ModuleList([Attention_Block(dim, n_heads, ffn) for _ in range(9*n_layers)])
        self.classifier = nn.Linear(dim*6*n_layers, 9, bias=False)  # dim*6*n_layers  dim
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
    
class Base_model(nn.Module):
    def __init__(self, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN):
        super().__init__()
        self.intensity = Multi_ATTN(dim, l_len, v_len, a_len, n_heads, n_layers, ffn)
        self.stimulation = Multi_ATTN(dim, l_len, v_len, a_len, n_heads, n_layers, ffn)
        self.trans = nn.Parameter(torch.rand(9,9,9), requires_grad=True)
        self.norm3 = nn.LayerNorm(9)
        self.out = nn.Linear(18, 9)
    def forward(self, pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask,\
                pro_video_feat, pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask):
        last_feat = self.intensity(pre_text_feat, pre_video_feat, pre_audio_feat, pre_text_mask, pre_video_mask, pre_audio_mask)
        this_feat = self.stimulation(pro_text_feat, pro_video_feat, pro_audio_feat, pro_text_mask, pro_video_mask, pro_audio_mask)
        batch_list = []
        for i in range(this_feat.shape[0]):
            temp_feat = torch.matmul(last_feat[i], self.trans)  # (7, 7)
            temp_feat = torch.matmul(this_feat[i], temp_feat)  # (7,)
            batch_list.append(temp_feat.unsqueeze(0))
        out_feat = torch.cat(batch_list, dim=0)  # (batch, 7)
        out_feat = torch.cat([this_feat, self.norm3(out_feat)], dim=1)
        return self.out(out_feat)

# loss
def multi_loss(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1], dtype = torch.float)
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

# trainer
def train(model, iterator, optimizer):
    model.train()
    epoch_loss, count = 0, 0
    iter_bar = tqdm(iterator, desc='Training')
    for _, batch in enumerate(iter_bar):
        count += 1
        optimizer.zero_grad()
        pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
        pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, label = zip(*batch)
        pre_text_feat = torch.FloatTensor(pre_text_feat).to(device)
        pre_text_mask = torch.FloatTensor(pre_text_mask).to(device)
        pro_text_feat = torch.FloatTensor(pro_text_feat).to(device)
        pro_text_mask = torch.FloatTensor(pro_text_mask).to(device)
        pre_video_feat = torch.FloatTensor(pre_video_feat).to(device)
        pre_video_mask = torch.FloatTensor(pre_video_mask).to(device)
        pro_video_feat = torch.FloatTensor(pro_video_feat).to(device)
        pro_video_mask = torch.FloatTensor(pro_video_mask).to(device)
        pre_audio_feat = torch.FloatTensor(pre_audio_feat).to(device)
        pre_audio_mask = torch.FloatTensor(pre_audio_mask).to(device)
        pro_audio_feat = torch.FloatTensor(pro_audio_feat).to(device)
        pro_audio_mask = torch.FloatTensor(pro_audio_mask).to(device)
        label = torch.FloatTensor(label).to(device)
        logits = model(pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask,\
                pro_video_feat, pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask)
        m_loss = multi_loss(logits, label)
        kl_0 = F.kl_div(F.logsigmoid(logits[::2]), torch.sigmoid(logits[1::2]), reduction='batchmean')
        kl_1 = F.kl_div(F.logsigmoid(logits[1::2]), torch.sigmoid(logits[::2]), reduction='batchmean')
        loss = m_loss + (kl_0+kl_1) / 2
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
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
            pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
            pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, label = zip(*batch)
            pre_text_feat = torch.FloatTensor(pre_text_feat).to(device)
            pre_text_mask = torch.FloatTensor(pre_text_mask).to(device)
            pro_text_feat = torch.FloatTensor(pro_text_feat).to(device)
            pro_text_mask = torch.FloatTensor(pro_text_mask).to(device)
            pre_video_feat = torch.FloatTensor(pre_video_feat).to(device)
            pre_video_mask = torch.FloatTensor(pre_video_mask).to(device)
            pro_video_feat = torch.FloatTensor(pro_video_feat).to(device)
            pro_video_mask = torch.FloatTensor(pro_video_mask).to(device)
            pre_audio_feat = torch.FloatTensor(pre_audio_feat).to(device)
            pre_audio_mask = torch.FloatTensor(pre_audio_mask).to(device)
            pro_audio_feat = torch.FloatTensor(pro_audio_feat).to(device)
            pro_audio_mask = torch.FloatTensor(pro_audio_mask).to(device)
            label = torch.FloatTensor(label).to(device)
            logits = model(pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask,\
                    pro_video_feat, pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask)
            loss = multi_loss(logits, label)
            iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss / count

def run(model, train_list, valid_list, batch_size, learning_rate, epochs, name):
    log_file = log_path+name+'.txt'
    with open(log_file, 'w') as log_f:
        log_f.write('epoch, train_loss, valid_loss\n')
    writer = SummaryWriter(log_path)
    my_list = ['norm.weight','norm.bias','classifier.weight','classifier.bias']
    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
    optimizer = optim.AdamW([{'params': base_params, 'lr': learning_rate}, {'params': params, 'lr': learning_rate}])
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(epochs):
        print('Epoch: ' + str(epoch+1))
        train_iterator = data_loader(train_list, batch_size)
        valid_iterator = data_loader(valid_list, batch_size)
        train_loss = train(model, train_iterator, optimizer)
        valid_loss = valid(model, valid_iterator)
        writer.add_scalars(name, {'train_loss':train_loss, 'valid_loss':valid_loss}, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss) 
        with open(log_file, 'a') as log_f:
            log_f.write('\n{epoch}, {train_loss: 3.3f}, {valid_loss: 3.3f}\n'.format(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss))
        if valid_loss == min(loss_list) and valid_loss > 0.009:
            stop = 0
            torch.save(model.state_dict(), os.path.join(log_path, name+'_'+str(valid_loss)[:4]+'.pt'))
        else:
            stop += 1
            if stop >= 3:
                break
    writer.close()
    
# train
random.shuffle(train_set)
model_1 = Base_model().to(device)
valid_list = train_set[:744]
train_list = train_set[744:]
run(model_1, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='base_5')

model_2 = Base_model().to(device)
valid_list = train_set[744:744*2]
train_list = train_set[:744] + train_set[744*2:]
run(model_2, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='base_6')

model_3 = Base_model().to(device)
valid_list = train_set[744*2:744*3]
train_list = train_set[:744*2] + train_set[744*3:]
run(model_3, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='base_7')

model_4 = Base_model().to(device)
valid_list = train_set[744*3:744*4]
train_list = train_set[:744*3] + train_set[744*4:]
run(model_4, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='base_8')

# evaluation
def data_loader(data_set, batch_size):
    count = 0
    while count < len(data_set):
        batch = []
        size = min(batch_size, len(data_set) - count)
        for _ in range(size):
            pro_name = data_set[count][0]
            name_list = pro_name.split('_')
            if name_list[-1] == '1':
                pre_name = pro_name
            else:
                name_list[-1] = str(int(name_list[-1])-1)
                pre_name = '_'.join(name_list)
            pre_text_feat, pre_text_mask = text_features(pre_name)
            pro_text_feat, pro_text_mask = text_features(pro_name)
            pre_video_feat, pre_video_mask = video_features(pre_name)
            pro_video_feat, pro_video_mask = video_features(pro_name)
            pre_audio_feat, pre_audio_mask = audio_features(pre_name)
            pro_audio_feat, pro_audio_mask = audio_features(pro_name)
            batch.append((pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
                          pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, data_set[count][1]))
            count += 1
        yield batch

def outputs():
    pred_1, pred_2, pred_3, pred_4, label_1 = [], [], [], [], []
    
    model_1 = Base_model().to(device)
    model_1.load_state_dict(torch.load(log_path + 'base_4_2.54.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(train_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
            pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, label = zip(*batch)
            pre_text_feat = torch.FloatTensor(pre_text_feat).to(device)
            pre_text_mask = torch.FloatTensor(pre_text_mask).to(device)
            pro_text_feat = torch.FloatTensor(pro_text_feat).to(device)
            pro_text_mask = torch.FloatTensor(pro_text_mask).to(device)
            pre_video_feat = torch.FloatTensor(pre_video_feat).to(device)
            pre_video_mask = torch.FloatTensor(pre_video_mask).to(device)
            pro_video_feat = torch.FloatTensor(pro_video_feat).to(device)
            pro_video_mask = torch.FloatTensor(pro_video_mask).to(device)
            pre_audio_feat = torch.FloatTensor(pre_audio_feat).to(device)
            pre_audio_mask = torch.FloatTensor(pre_audio_mask).to(device)
            pro_audio_feat = torch.FloatTensor(pro_audio_feat).to(device)
            pro_audio_mask = torch.FloatTensor(pro_audio_mask).to(device)
            label = torch.FloatTensor(label)
            logits = model_1(pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask,\
                    pro_video_feat, pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask).detach().cpu()
            pred_1.append(logits)
            label_1.append(label)
    pred_1 = torch.cat(pred_1, dim=0)
    label_1 = torch.cat(label_1, dim=0)
    
    model_1 = Base_model().to(device)
    model_1.load_state_dict(torch.load(log_path + 'base_3_2.52.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(train_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
            pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, label = zip(*batch)
            pre_text_feat = torch.FloatTensor(pre_text_feat).to(device)
            pre_text_mask = torch.FloatTensor(pre_text_mask).to(device)
            pro_text_feat = torch.FloatTensor(pro_text_feat).to(device)
            pro_text_mask = torch.FloatTensor(pro_text_mask).to(device)
            pre_video_feat = torch.FloatTensor(pre_video_feat).to(device)
            pre_video_mask = torch.FloatTensor(pre_video_mask).to(device)
            pro_video_feat = torch.FloatTensor(pro_video_feat).to(device)
            pro_video_mask = torch.FloatTensor(pro_video_mask).to(device)
            pre_audio_feat = torch.FloatTensor(pre_audio_feat).to(device)
            pre_audio_mask = torch.FloatTensor(pre_audio_mask).to(device)
            pro_audio_feat = torch.FloatTensor(pro_audio_feat).to(device)
            pro_audio_mask = torch.FloatTensor(pro_audio_mask).to(device)
            label = torch.FloatTensor(label)
            logits = model_1(pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask,\
                    pro_video_feat, pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask).detach().cpu()
            pred_2.append(logits)
    pred_2 = torch.cat(pred_2, dim=0)
    
    model_1 = Base_model().to(device)
    model_1.load_state_dict(torch.load(log_path + 'base_2_2.55.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(train_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
            pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, label = zip(*batch)
            pre_text_feat = torch.FloatTensor(pre_text_feat).to(device)
            pre_text_mask = torch.FloatTensor(pre_text_mask).to(device)
            pro_text_feat = torch.FloatTensor(pro_text_feat).to(device)
            pro_text_mask = torch.FloatTensor(pro_text_mask).to(device)
            pre_video_feat = torch.FloatTensor(pre_video_feat).to(device)
            pre_video_mask = torch.FloatTensor(pre_video_mask).to(device)
            pro_video_feat = torch.FloatTensor(pro_video_feat).to(device)
            pro_video_mask = torch.FloatTensor(pro_video_mask).to(device)
            pre_audio_feat = torch.FloatTensor(pre_audio_feat).to(device)
            pre_audio_mask = torch.FloatTensor(pre_audio_mask).to(device)
            pro_audio_feat = torch.FloatTensor(pro_audio_feat).to(device)
            pro_audio_mask = torch.FloatTensor(pro_audio_mask).to(device)
            label = torch.FloatTensor(label)
            logits = model_1(pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask,\
                    pro_video_feat, pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask).detach().cpu()
            pred_3.append(logits)
    pred_3 = torch.cat(pred_3, dim=0)
    
    model_1 = Base_model().to(device)
    model_1.load_state_dict(torch.load(log_path + 'base_1_2.47.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(train_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
            pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, label = zip(*batch)
            pre_text_feat = torch.FloatTensor(pre_text_feat).to(device)
            pre_text_mask = torch.FloatTensor(pre_text_mask).to(device)
            pro_text_feat = torch.FloatTensor(pro_text_feat).to(device)
            pro_text_mask = torch.FloatTensor(pro_text_mask).to(device)
            pre_video_feat = torch.FloatTensor(pre_video_feat).to(device)
            pre_video_mask = torch.FloatTensor(pre_video_mask).to(device)
            pro_video_feat = torch.FloatTensor(pro_video_feat).to(device)
            pro_video_mask = torch.FloatTensor(pro_video_mask).to(device)
            pre_audio_feat = torch.FloatTensor(pre_audio_feat).to(device)
            pre_audio_mask = torch.FloatTensor(pre_audio_mask).to(device)
            pro_audio_feat = torch.FloatTensor(pro_audio_feat).to(device)
            pro_audio_mask = torch.FloatTensor(pro_audio_mask).to(device)
            label = torch.FloatTensor(label)
            logits = model_1(pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask,\
                    pro_video_feat, pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask).detach().cpu()
            pred_4.append(logits)
    pred_4 = torch.cat(pred_4, dim=0)

    return pred_1+pred_2+pred_3+pred_4, label_1

pred, label = outputs()

import math
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, coverage_error, label_ranking_loss, label_ranking_average_precision_score

def sigmoids(x, t):
    return 1/(1 + math.exp(-x+t))

def test(pred, label):
    zero = torch.zeros_like(pred)
    one = torch.ones_like(pred)

    label_all, pred_all = [], []
    for j in range(label.shape[0]):
        label_all.append(label[j][:8].int().tolist())
    for j in range(pred.shape[0]):
        pred_all.append(pred[j][:8].float().tolist())
        
    temp_max = 0.0

    for love in [-3.6]:
        for anxi in [-1.2]:
            for sorr in [-1.4]:
                for joyy in [-3.4]:
                    for expe in [-2.0]:
                        for hate in [-1.4]:
                            for ange in [-2.6]:
                                for surp in [-3.8]:
                                    sigm_all, bina_all = [], []
                                    bina_love = torch.where(pred > (love), one, zero)
                                    bina_anxi = torch.where(pred > (anxi), one, zero)
                                    bina_sorr = torch.where(pred > (sorr), one, zero)
                                    bina_joyy = torch.where(pred > (joyy), one, zero)
                                    bina_expe = torch.where(pred > (expe), one, zero)
                                    bina_hate = torch.where(pred > (hate), one, zero)
                                    bina_ange = torch.where(pred > (ange), one, zero)
                                    bina_surp = torch.where(pred > (surp), one, zero)
                                    for j in range(pred.shape[0]):
                                        temp_bina = [0,0,0,0,0,0,0,0]
                                        temp_bina[0] = int(bina_love[j][0])
                                        temp_bina[1] = int(bina_anxi[j][1])
                                        temp_bina[2] = int(bina_sorr[j][2])
                                        temp_bina[3] = int(bina_joyy[j][3])
                                        temp_bina[4] = int(bina_expe[j][4])
                                        temp_bina[5] = int(bina_hate[j][5])
                                        temp_bina[6] = int(bina_ange[j][6])
                                        temp_bina[7] = int(bina_surp[j][7])
                                        bina_all.append(temp_bina)
                                    f1 = f1_score(label_all, bina_all, average='micro')+f1_score(label_all, bina_all, average='macro')
                                    if f1 > temp_max:
                                        temp_max = f1
                                        print(love, anxi, sorr, joyy, expe, hate, ange, surp, f1)
    return 0

test(pred, label)

def outputs():
    pred_1, pred_2, pred_3, pred_4, label_1 = [], [], [], [], []
    
    model_1 = Base_model().to(device)
    model_1.load_state_dict(torch.load(log_path + 'base_4_2.54.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
            pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, label = zip(*batch)
            pre_text_feat = torch.FloatTensor(pre_text_feat).to(device)
            pre_text_mask = torch.FloatTensor(pre_text_mask).to(device)
            pro_text_feat = torch.FloatTensor(pro_text_feat).to(device)
            pro_text_mask = torch.FloatTensor(pro_text_mask).to(device)
            pre_video_feat = torch.FloatTensor(pre_video_feat).to(device)
            pre_video_mask = torch.FloatTensor(pre_video_mask).to(device)
            pro_video_feat = torch.FloatTensor(pro_video_feat).to(device)
            pro_video_mask = torch.FloatTensor(pro_video_mask).to(device)
            pre_audio_feat = torch.FloatTensor(pre_audio_feat).to(device)
            pre_audio_mask = torch.FloatTensor(pre_audio_mask).to(device)
            pro_audio_feat = torch.FloatTensor(pro_audio_feat).to(device)
            pro_audio_mask = torch.FloatTensor(pro_audio_mask).to(device)
            label = torch.FloatTensor(label)
            logits = model_1(pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask,\
                    pro_video_feat, pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask).detach().cpu()
            pred_1.append(logits)
            label_1.append(label)
    pred_1 = torch.cat(pred_1, dim=0)
    label_1 = torch.cat(label_1, dim=0)
    
    model_1 = Base_model().to(device)
    model_1.load_state_dict(torch.load(log_path + 'base_3_2.52.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
            pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, label = zip(*batch)
            pre_text_feat = torch.FloatTensor(pre_text_feat).to(device)
            pre_text_mask = torch.FloatTensor(pre_text_mask).to(device)
            pro_text_feat = torch.FloatTensor(pro_text_feat).to(device)
            pro_text_mask = torch.FloatTensor(pro_text_mask).to(device)
            pre_video_feat = torch.FloatTensor(pre_video_feat).to(device)
            pre_video_mask = torch.FloatTensor(pre_video_mask).to(device)
            pro_video_feat = torch.FloatTensor(pro_video_feat).to(device)
            pro_video_mask = torch.FloatTensor(pro_video_mask).to(device)
            pre_audio_feat = torch.FloatTensor(pre_audio_feat).to(device)
            pre_audio_mask = torch.FloatTensor(pre_audio_mask).to(device)
            pro_audio_feat = torch.FloatTensor(pro_audio_feat).to(device)
            pro_audio_mask = torch.FloatTensor(pro_audio_mask).to(device)
            label = torch.FloatTensor(label)
            logits = model_1(pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask,\
                    pro_video_feat, pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask).detach().cpu()
            pred_2.append(logits)
    pred_2 = torch.cat(pred_2, dim=0)
    
    model_1 = Base_model().to(device)
    model_1.load_state_dict(torch.load(log_path + 'base_2_2.55.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
            pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, label = zip(*batch)
            pre_text_feat = torch.FloatTensor(pre_text_feat).to(device)
            pre_text_mask = torch.FloatTensor(pre_text_mask).to(device)
            pro_text_feat = torch.FloatTensor(pro_text_feat).to(device)
            pro_text_mask = torch.FloatTensor(pro_text_mask).to(device)
            pre_video_feat = torch.FloatTensor(pre_video_feat).to(device)
            pre_video_mask = torch.FloatTensor(pre_video_mask).to(device)
            pro_video_feat = torch.FloatTensor(pro_video_feat).to(device)
            pro_video_mask = torch.FloatTensor(pro_video_mask).to(device)
            pre_audio_feat = torch.FloatTensor(pre_audio_feat).to(device)
            pre_audio_mask = torch.FloatTensor(pre_audio_mask).to(device)
            pro_audio_feat = torch.FloatTensor(pro_audio_feat).to(device)
            pro_audio_mask = torch.FloatTensor(pro_audio_mask).to(device)
            label = torch.FloatTensor(label)
            logits = model_1(pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask,\
                    pro_video_feat, pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask).detach().cpu()
            pred_3.append(logits)
    pred_3 = torch.cat(pred_3, dim=0)
    
    model_1 = Base_model().to(device)
    model_1.load_state_dict(torch.load(log_path + 'base_1_2.47.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask, pro_video_feat,\
            pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask, label = zip(*batch)
            pre_text_feat = torch.FloatTensor(pre_text_feat).to(device)
            pre_text_mask = torch.FloatTensor(pre_text_mask).to(device)
            pro_text_feat = torch.FloatTensor(pro_text_feat).to(device)
            pro_text_mask = torch.FloatTensor(pro_text_mask).to(device)
            pre_video_feat = torch.FloatTensor(pre_video_feat).to(device)
            pre_video_mask = torch.FloatTensor(pre_video_mask).to(device)
            pro_video_feat = torch.FloatTensor(pro_video_feat).to(device)
            pro_video_mask = torch.FloatTensor(pro_video_mask).to(device)
            pre_audio_feat = torch.FloatTensor(pre_audio_feat).to(device)
            pre_audio_mask = torch.FloatTensor(pre_audio_mask).to(device)
            pro_audio_feat = torch.FloatTensor(pro_audio_feat).to(device)
            pro_audio_mask = torch.FloatTensor(pro_audio_mask).to(device)
            label = torch.FloatTensor(label)
            logits = model_1(pre_text_feat, pre_text_mask, pro_text_feat, pro_text_mask, pre_video_feat, pre_video_mask,\
                    pro_video_feat, pro_video_mask, pre_audio_feat, pre_audio_mask, pro_audio_feat, pro_audio_mask).detach().cpu()
            pred_4.append(logits)
    pred_4 = torch.cat(pred_4, dim=0)

    return pred_1+pred_2+pred_3+pred_4, label_1

pred, label = outputs()

def test(pred, label):
    zero = torch.zeros_like(pred)
    one = torch.ones_like(pred)
    
    pred_love = torch.where(pred > (-3.6), one, zero)  # -4.0
    pred_anxi = torch.where(pred > (-1.2), one, zero)  # -1.6
    pred_sorr = torch.where(pred > (-1.4), one, zero)  # 
    pred_joyy = torch.where(pred > (-3.4), one, zero)  # 
    pred_expe = torch.where(pred > (-2.0), one, zero)  # 
    pred_hate = torch.where(pred > (-1.4), one, zero)  # 
    pred_ange = torch.where(pred > (-2.6), one, zero)  # 
    pred_surp = torch.where(pred > (-3.8), one, zero)  # 
    
    label_love, soft_love = [], []
    label_anxi, soft_anxi = [], []
    label_sorr, soft_sorr = [], []
    label_joyy, soft_joyy = [], []
    label_expe, soft_expe = [], []
    label_hate, soft_hate = [], []
    label_ange, soft_ange = [], []
    label_surp, soft_surp = [], []
    
    for i in range(len(label)):
        label_love.append(int(label[i][0]))
        label_anxi.append(int(label[i][1]))
        label_sorr.append(int(label[i][2]))
        label_joyy.append(int(label[i][3]))
        label_expe.append(int(label[i][4]))
        label_hate.append(int(label[i][5]))
        label_ange.append(int(label[i][6]))
        label_surp.append(int(label[i][7]))

        soft_love.append(int(pred_love[i][0]))
        soft_anxi.append(int(pred_anxi[i][1]))
        soft_sorr.append(int(pred_sorr[i][2]))
        soft_joyy.append(int(pred_joyy[i][3]))
        soft_expe.append(int(pred_expe[i][4]))
        soft_hate.append(int(pred_hate[i][5]))
        soft_ange.append(int(pred_ange[i][6]))
        soft_surp.append(int(pred_surp[i][7]))
            
    print('love_acc: ', accuracy_score(label_love, soft_love))
    print('love_f1: ', f1_score(label_love, soft_love, average='weighted'))
    print('anxi_acc: ', accuracy_score(label_anxi, soft_anxi))
    print('anxi_f1: ', f1_score(label_anxi, soft_anxi, average='weighted'))
    print('sorr_acc: ', accuracy_score(label_sorr, soft_sorr))
    print('sorr_f1: ', f1_score(label_sorr, soft_sorr, average='weighted'))
    print('joyy_acc: ', accuracy_score(label_joyy, soft_joyy))
    print('joyy_f1: ', f1_score(label_joyy, soft_joyy, average='weighted'))
    print('expe_acc: ', accuracy_score(label_expe, soft_expe))
    print('expe_f1: ', f1_score(label_expe, soft_expe, average='weighted'))
    print('hate_acc: ', accuracy_score(label_hate, soft_hate))
    print('hate_f1: ', f1_score(label_hate, soft_hate, average='weighted'))
    print('ange_acc: ', accuracy_score(label_ange, soft_ange))
    print('ange_f1: ', f1_score(label_ange, soft_ange, average='weighted'))
    print('surp_acc: ', accuracy_score(label_surp, soft_surp))
    print('surp_f1: ', f1_score(label_surp, soft_surp, average='weighted'))
    return 0

test(pred, label)
