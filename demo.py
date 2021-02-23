# import
import os
import math
import random
import pickle
import numpy as np
from tqdm import tqdm
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
video_file = '/home/dango/multimodal/CMU-MOSEI/align/Feature(0)-360/'
audio_file = '/home/dango/multimodal/CMU-MOSEI/align/WAV_feature/'
text_file = '/home/dango/multimodal/CMU-MOSEI/align/TXT_feature/'  # mosei
label_file = '/home/dango/multimodal/CMU-MOSEI/align/labels.txt'
ren_feat_file = '/home/dango/multimodal/ren/ren_text_feat/'
ren_txt_file = '/home/dango/multimodal/ren/1487_txt_hier_sents_202002/'
ren_xml_file = '/home/dango/multimodal/ren/1487_xml_doc_segmented_utf8/'
log_dir = '/home/dango/multimodal/CMU-MOSEI/par_log/demo_1/'

# 参数
EPOCHS = 99
CLIP = 1.0
LR = 0.001
L_LEN = 25
V_LEN = 100
A_LEN = 100
DIM = 192
BATCH = 64
DROP = 0.1
FFN = 2
N_HEADS = 6
N_LAYERS = 2

# 数据名列表
name_list = os.listdir(video_file)
for i in range(len(name_list)):
    name_list[i] = name_list[i].split('.pk')[0]

# 数据名-标签字典
def label_dictionary(file_name, name_list):    # happ sadn ange disg surp fear neut
    dictionary = {}
    with open(file_name, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            if line.split(',')[0] in name_list:
                dictionary[line.split(',')[0]] = line.strip().split(',')[3:]
    return dictionary

label_dict = label_dictionary(label_file, name_list)

# 视频特征 + mask
def video_features(file_name, name, v_len):  # 0 ~ 2309
    with open(file_name + name + '.pk', 'rb') as f:
        feat_list = pickle.load(f)
    if len(feat_list) == 0:
        feat_1024 = np.zeros((v_len, 1024))
        feat_512 = np.zeros((v_len, 512))
        feat_256 = np.zeros((v_len, 256))
        mask = np.zeros(v_len)
    else:
        list_1024, list_512, list_256, temp_list = [], [], [], []
        for i in range(len(feat_list)):
            if feat_list[i].shape[0] == 1024:
                list_1024.append(feat_list[i])
            elif feat_list[i].shape[0] == 512:
                list_512.append(feat_list[i])
            elif feat_list[i].shape[0] == 256:
                list_256.append(feat_list[i])
        if len(list_1024) >= len(list_512) and len(list_1024) >= len(list_256):
            feat_list = list_1024
        elif len(list_512) >= len(list_1024) and len(list_512) >= len(list_256):
            feat_list = list_512
        elif len(list_256) >= len(list_1024) and len(list_256) >= len(list_512):
            feat_list = list_256
        if len(feat_list) < v_len:
            for i in range(len(feat_list)):
                temp_list.append(np.expand_dims(feat_list[i], axis=0))
            pad_len = v_len - len(feat_list)
            for i in range(pad_len):
                temp_list.append(np.zeros((1, feat_list[0].shape[0])))
            feat = np.concatenate(temp_list, axis=0)
            mask = np.concatenate((np.ones(len(feat_list)), np.zeros(pad_len)), axis=0)
        else:
            gap = len(feat_list) // v_len
            for i in range(0, len(feat_list), gap):
                temp_list.append(np.expand_dims(feat_list[i], axis=0))
            feat = np.concatenate(temp_list[:v_len], axis=0)
            mask = np.ones(v_len)
        if feat.shape[1] == 1024:
            feat_1024 = feat
            feat_512 = np.zeros((v_len, 512))
            feat_256 = np.zeros((v_len, 256))
        elif feat.shape[1] == 512:
            feat_1024 = np.zeros((v_len, 1024))
            feat_512 = feat
            feat_256 = np.zeros((v_len, 256))
        elif feat.shape[1] == 256:
            feat_1024 = np.zeros((v_len, 1024))
            feat_512 = np.zeros((v_len, 512))
            feat_256 = feat
    return feat_256, feat_512, feat_1024, mask

# 音频特征 + mask
def audio_features(file_name, name, a_len):  # 10 ~ 2417  avg 252
    temp_feat = np.load(file_name + name + '.npy')
    temp_list = []
    if len(temp_feat) == 0:
        feat = np.zeros((a_len, 40))
        mask = np.zeros(a_len)
    elif len(temp_feat) < a_len:
        temp_list.append(temp_feat)
        pad_len = a_len - len(temp_feat)
        temp_list.append(np.zeros((pad_len, 40)))
        feat = np.concatenate(temp_list, axis=0)
        mask = np.concatenate((np.ones(len(temp_feat)), np.zeros(pad_len)), axis=0)
    else:
        gap = len(temp_feat) // a_len
        for i in range(0, len(temp_feat), gap):
            temp_list.append(np.expand_dims(temp_feat[i], axis=0))
        feat = np.concatenate(temp_list[:a_len], axis=0)
        mask = np.ones(a_len)
    return feat, mask

# 文本特征 + mask
def text_features(file_name, name, l_len):  # 4 ~ 110 mosei   1 ~ 502 ren  avg 24
    temp_feat = np.load(file_name + name + '.npy')
    temp_list = []
    if len(temp_feat) == 0:
        feat = np.zeros((l_len, 768))
        mask = np.zeros(l_len)
    elif len(temp_feat) < l_len:
        temp_list.append(temp_feat)
        pad_len = l_len - len(temp_feat)
        temp_list.append(np.zeros((pad_len, 768)))
        feat = np.concatenate(temp_list, axis=0)
        mask = np.concatenate((np.ones(len(temp_feat)), np.zeros(pad_len)), axis=0)
    else:
        gap = len(temp_feat) // l_len
        for i in range(0, len(temp_feat), gap):
            temp_list.append(np.expand_dims(temp_feat[i], axis=0))
        feat = np.concatenate(temp_list[:l_len], axis=0)
        mask = np.ones(l_len)
    return feat, mask

# 判断中文
def check_contain_chinese(strings):
    chinese = False
    for i in range(len(strings)):
        if u'\u4e00' <= strings[i] <= u'\u9fff':
            chinese = True
    return chinese

# 将ren_label替换为mosei_label
def ren_label_list(num):
    label_list, count_list = [], []
    with open(ren_txt_file + 'cet_' + str(num) + '.txt', 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            if line[0] == 's':
                count += 1
                if line.split(':')[2] == '\n' or line.split(':')[2] == '/n\n' or line.split(':')[2] == '/n' or line.split(':')[2] == '' or line.split(':')[2][0] == '/':
                    count_list.append(count)
                    continue
                else:
                    temp_text = line.strip().split(':')[2].split('  ')
                    for i in range(len(temp_text)):
                        temp_text[i] = temp_text[i].split('/')[0]
                    if check_contain_chinese(temp_text) == False:
                        count_list.append(count)
                        continue
                    else:
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
    return label_list, count_list

# ren数据名
def ren_para_sent_list(num, count_list):
    para_list, sent_list = [], []
    with open(ren_xml_file + 'cet_' + str(num) + '.xml', 'r') as f:
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

# 标签对应ren数据名
def ren_label_name():
    label_list, name_list = [], []
    for i in range(1, 1488):
        if i == 490 or i == 761:
            continue
        labe_list, count_list = ren_label_list(i)
        para_list, sent_list = ren_para_sent_list(i, count_list)
        for j in range(len(para_list)):
            name_list.append(str(i)+'_'+para_list[j]+'_'+sent_list[j])
            label_list.append(labe_list[j])
    return label_list, name_list

# ren标签数据名字典
def ren_label_name_dict():
    dicts = {}
    label_list, name_list = ren_label_name()
    for j in range(len(label_list)):
        if label_list[j] not in dicts.keys():
            temp_list = [name_list[j]]
            dicts[label_list[j]] = temp_list
        else:
            temp_list = dicts[label_list[j]]
            temp_list.append(name_list[j])
            dicts[label_list[j]] = temp_list
    dict_keys = dicts.keys()  # 随机注释
    for key in dict_keys:
        temp_list = dicts[key]
        random.shuffle(temp_list)
        dicts[key] = temp_list
    return dicts

# 数据加载器
def data_loader(name_list, batch_size):
    random.shuffle(name_list)   # 随机注释
    replace_dict = ren_label_name_dict()
    count = 0
    while count < len(name_list):
        batch = []
        size = min(batch_size, len(name_list) - count)
        for _ in range(size):
            label = label_dict[name_list[count]]
            for i in range(len(label)):
                label[i] = str(label[i])
            if ''.join(label) in replace_dict.keys():
                replace_list = replace_dict[''.join(label)]
                replace_name = replace_list[0]
                replace_list.append(replace_name)
                replace_dict[''.join(label)] = replace_list[1:]
            else:
                replace_list = replace_dict['0000001']
                replace_name = replace_list[0]
                replace_list.append(replace_name)
                replace_dict['0000001'] = replace_list[1:]
            l, l_mask = text_features(ren_feat_file, replace_name, L_LEN)
            v_256, v_512, v_1024, v_mask = video_features(video_file, name_list[count], V_LEN)
            a, a_mask = audio_features(audio_file, name_list[count], A_LEN)
            for i in range(len(label)):
                label[i] = int(label[i])
            batch.append((l, v_256, v_512, v_1024, a, l_mask, v_mask, a_mask, label))
            count += 1
        yield batch

# 计算参数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# 统一 hidden size
class Unify_Dimension_Conv1d(nn.Module):
    def __init__(self, dim, l_dim=768, dim_1024=1024, dim_512=512, dim_256=256, a_dim=40):
        super().__init__()
        self.linguistic = nn.Conv1d(l_dim, dim, kernel_size=1)
        self.visual_1024 = nn.Conv1d(dim_1024, dim//3, kernel_size=1)
        self.visual_512 = nn.Conv1d(dim_512, dim//3, kernel_size=1)
        self.visual_256 = nn.Conv1d(dim_256, dim//3, kernel_size=1)
        self.acoustic = nn.Conv1d(a_dim, dim, kernel_size=1)
        self.drop = nn.Dropout(DROP)
    def forward(self, l, v_256, v_512, v_1024, a):
        l, v_256, v_512, v_1024, a = l.transpose(1, 2), v_256.transpose(1, 2), v_512.transpose(1, 2), v_1024.transpose(1, 2), a.transpose(1, 2)
        l = self.drop(self.linguistic(l))
        v_256 = self.drop(self.visual_256(v_256))
        v_512 = self.drop(self.visual_512(v_512))
        v_1024 = self.drop(self.visual_1024(v_1024))
        a = self.drop(self.acoustic(a))
        v_256, v_512, v_1024 = v_256.transpose(1, 2), v_512.transpose(1, 2), v_1024.transpose(1, 2)
        v = torch.cat((v_256, v_512, v_1024), 2)
        return l.transpose(1, 2), v, a.transpose(1, 2)

# 位置信息
class Position_Embedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, dim)
        self.len = max_len
    def forward(self, x):
        position_ids = torch.arange(self.len, device=device).unsqueeze(0).repeat(x.size()[0],1)
        return self.position_embeddings(position_ids.to(device))

# 注意力
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

# 多分类模型
class Multi_class(nn.Module):
    def __init__(self, dim, l_len, v_len, a_len, n_heads, n_layers, ffn):
        super().__init__()
        self.unify_dimension = Unify_Dimension_Conv1d(dim)
        self.linguistic_position = Position_Embedding(l_len, dim)
        self.visual_position = Position_Embedding(v_len, dim)
        self.acoustic_position = Position_Embedding(a_len, dim)
        self.n_layers = n_layers
        self.multimodal_blocks = nn.ModuleList([Attention_Block(dim, n_heads, ffn) for _ in range(9*n_layers)])
        self.fully_connected = nn.Linear(dim*6, dim)
        self.normalization = nn.LayerNorm(dim)
        self.drop = nn.Dropout(DROP)
        self.classifier = nn.Linear(dim*6*n_layers, 7)  # dim*6*n_layers  dim
    def forward(self, l, v_256, v_512, v_1024, a, l_mask, v_mask, a_mask):
        l, v, a = self.unify_dimension(l, v_256, v_512, v_1024, a)
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
#         x = self.drop(nn.ReLU()(self.normalization(self.fully_connected(x))))  # (batch, dim)
        return self.classifier(x)

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
        linguistic, visual_256, visual_512, visual_1024, acoustic, l_mask, v_mask, a_mask, label = zip(*batch)
        linguistic, visual_256, visual_512, visual_1024, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual_256),\
        torch.cuda.FloatTensor(visual_512), torch.cuda.FloatTensor(visual_1024), torch.cuda.FloatTensor(acoustic),torch.cuda.FloatTensor(l_mask),\
        torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(label)
        logits_clsf = model(linguistic, visual_256, visual_512, visual_1024, acoustic, l_mask, v_mask, a_mask)
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
            linguistic, visual_256, visual_512, visual_1024, acoustic, l_mask, v_mask, a_mask, label = zip(*batch)
            linguistic, visual_256, visual_512, visual_1024, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual_256),\
            torch.cuda.FloatTensor(visual_512), torch.cuda.FloatTensor(visual_1024), torch.cuda.FloatTensor(acoustic),torch.cuda.FloatTensor(l_mask),\
            torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(label)
            logits_clsf = model(linguistic, visual_256, visual_512, visual_1024, acoustic, l_mask, v_mask, a_mask)
            loss = multi_circle_loss(logits_clsf, label)
            loss = loss.mean()
            iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss, count, epoch_loss / count

# 训练器
def run(model, train_list, valid_list, batch_size, learning_rate, epochs, log_name):
    log_file = log_dir+log_name+'.txt'
    with open(log_file, 'w') as log_f:
        log_f.write('epoch, train_loss, valid_loss\n')
    writer = SummaryWriter(log_dir)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(epochs):
        print('Epoch: ' + str(epoch+1))
        train_iterator = data_loader(train_list, batch_size)
        valid_iterator = data_loader(valid_list, batch_size)
        train_loss = train(model, train_iterator, optimizer)
        _, _, valid_loss = valid(model, valid_iterator)
        writer.add_scalars(log_name, {'train_loss':train_loss, 'valid_loss':valid_loss}, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss) 
        with open(log_file, 'a') as log_f:
            log_f.write('\n{epoch},{train_loss: 2.2f},{valid_loss: 2.2f}\n'.format(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss))
        if valid_loss == min(loss_list):
            stop = 0
            torch.save(model.state_dict(), os.path.join(log_dir, log_name+'_'+str(valid_loss)[:4]+'.pt'))
        else:
            stop += 1
            if stop >= 7:
                break
    writer.close()

def f1_calculation(test_list, model_1, model_2, model_3, model_4):
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    with torch.no_grad():
        for i in tqdm(range(13)):
            t = i/10-1.0
            test_iterator = data_loader(test_list, batch_size=64)
            label_happ, soft_happ = [], []
            label_sadn, soft_sadn = [], []
            label_ange, soft_ange = [], []
            label_surp, soft_surp = [], []
            label_disg, soft_disg = [], []
            label_fear, soft_fear = [], []
            for _, batch in enumerate(test_iterator):
                linguistic, visual_256, visual_512, visual_1024, acoustic, l_mask, v_mask, a_mask, label = zip(*batch)
                linguistic, visual_256, visual_512, visual_1024, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual_256),\
                torch.cuda.FloatTensor(visual_512), torch.cuda.FloatTensor(visual_1024), torch.cuda.FloatTensor(acoustic),torch.cuda.FloatTensor(l_mask),\
                torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(label)
                pred_1 = (model_1(linguistic, visual_256, visual_512, visual_1024, acoustic, l_mask, v_mask, a_mask)).detach().cpu()
                pred_2 = (model_2(linguistic, visual_256, visual_512, visual_1024, acoustic, l_mask, v_mask, a_mask)).detach().cpu()
                pred_3 = (model_3(linguistic, visual_256, visual_512, visual_1024, acoustic, l_mask, v_mask, a_mask)).detach().cpu()
                pred_4 = (model_4(linguistic, visual_256, visual_512, visual_1024, acoustic, l_mask, v_mask, a_mask)).detach().cpu()
                pred = (pred_1+pred_2+pred_3+pred_4) / 4
                zero = torch.zeros_like(pred)
                one = torch.ones_like(pred)
                label = label.detach().cpu()
                pred_t = torch.where(pred > t, one, zero)
                for j in range(len(label)):  # happ sadn ange disg surp fear neut
                    label_happ.append(int(label[j][0]))
                    label_sadn.append(int(label[j][1]))
                    label_ange.append(int(label[j][2]))
                    label_surp.append(int(label[j][4]))
                    label_disg.append(int(label[j][3]))
                    label_fear.append(int(label[j][5]))
                    soft_happ.append(int(pred_t[j][0]))
                    soft_sadn.append(int(pred_t[j][1]))
                    soft_ange.append(int(pred_t[j][2]))
                    soft_surp.append(int(pred_t[j][4]))
                    soft_disg.append(int(pred_t[j][3]))
                    soft_fear.append(int(pred_t[j][5]))
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
    return 0

model_1 = Multi_class(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_1.load_state_dict(torch.load(log_dir + 'model_1_1.31.pt'))
model_2 = Multi_class(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_2.load_state_dict(torch.load(log_dir + 'model_2_1.37.pt'))
model_3 = Multi_class(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_3.load_state_dict(torch.load(log_dir + 'model_3_1.39.pt'))
model_4 = Multi_class(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_4.load_state_dict(torch.load(log_dir + 'model_4_1.32.pt'))
f1_calculation(name_list, model_1, model_2, model_3, model_4)

def demo_output(video_file, v_name, audio_file, a_name, ren_feat_file, l_name, model_1, model_2, model_3, model_4):
    l, l_mask = text_features(ren_feat_file, l_name, L_LEN)
    l, l_mask = torch.cuda.FloatTensor(np.expand_dims(l, axis=0)), torch.cuda.FloatTensor(np.expand_dims(l_mask, axis=0))
    v_256, v_512, v_1024, v_mask = video_features(video_file, v_name, V_LEN)
    v_256, v_512 = torch.cuda.FloatTensor(np.expand_dims(v_256, axis=0)), torch.cuda.FloatTensor(np.expand_dims(v_512, axis=0))
    v_1024, v_mask = torch.cuda.FloatTensor(np.expand_dims(v_1024, axis=0)), torch.cuda.FloatTensor(np.expand_dims(v_mask, axis=0))
    a, a_mask = audio_features(audio_file, a_name, A_LEN)
    a, a_mask = torch.cuda.FloatTensor(np.expand_dims(a, axis=0)), torch.cuda.FloatTensor(np.expand_dims(a_mask, axis=0))
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    with torch.no_grad():
        pred_1 = (model_1(l, v_256, v_512, v_1024, a, l_mask, v_mask, a_mask)).detach().cpu()
        pred_2 = (model_2(l, v_256, v_512, v_1024, a, l_mask, v_mask, a_mask)).detach().cpu()
        pred_3 = (model_3(l, v_256, v_512, v_1024, a, l_mask, v_mask, a_mask)).detach().cpu()
        pred_4 = (model_4(l, v_256, v_512, v_1024, a, l_mask, v_mask, a_mask)).detach().cpu()
        pred = (pred_1+pred_2+pred_3+pred_4) / 4
        zero = torch.zeros_like(pred)
        one = torch.ones_like(pred)
        pred_happ = torch.where(pred > 0.0, one, zero)    # happ sadn ange disg surp fear neut
        pred_sadn = torch.where(pred > 0.1, one, zero)
        pred_ange = torch.where(pred > -0.1, one, zero)
        pred_disg = torch.where(pred > -0.1, one, zero)
        pred_surp = torch.where(pred > 0.0, one, zero)
        pred_fear = torch.where(pred > 0.0, one, zero)
        print('The emotion(s) is(are)')
        if int(pred_happ[0][0]) == 1:
            print('happy')
        if int(pred_sadn[0][1]) == 1:
            print('sad')
        if int(pred_ange[0][2]) == 1:
            print('angry')
        if int(pred_disg[0][3]) == 1:
            print('disgust')
        if int(pred_surp[0][4]) == 1:
            print('surprise')
        if int(pred_fear[0][5]) == 1:
            print('fear')
        if int(pred_happ[0][0])+int(pred_sadn[0][1])+int(pred_ange[0][2])+int(pred_disg[0][3])+int(pred_surp[0][4])+int(pred_fear[0][5]) == 0:
            print('neutral')

model_1 = Multi_class(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_1.load_state_dict(torch.load(log_dir + 'model_1_1.31.pt'))
model_2 = Multi_class(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_2.load_state_dict(torch.load(log_dir + 'model_2_1.37.pt'))
model_3 = Multi_class(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_3.load_state_dict(torch.load(log_dir + 'model_3_1.39.pt'))
model_4 = Multi_class(dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
model_4.load_state_dict(torch.load(log_dir + 'model_4_1.32.pt'))

v_name = '0jtdrsmVzYQ[0]'
a_name = '0jtdrsmVzYQ[0]'
l_name = '99_1_1'
demo_output(video_file, v_name, audio_file, a_name, ren_feat_file, l_name, model_1, model_2, model_3, model_4)

# The emotion(s) is(are)
# happy
# sad
