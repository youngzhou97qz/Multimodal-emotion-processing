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
save_dir = '/home/'+user+'/weight/'
log_dir = '/home/'+user+'/log/'
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
BATCH = 32
LR = 0.0001
CLIP = 1.0
EPOCHS = 99

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

weight = {'sentiment_7':torch.cuda.FloatTensor([18198/660, 18198/1743, 18198/2842, 18198/3971, 18198/5920, 18198/2507, 18198/555]),
          'sentiment_2':torch.cuda.FloatTensor([18198/5245, 18198/12953]),
          'happiness':torch.cuda.FloatTensor([18198/9740, 18198/8458]),
          'sadness':torch.cuda.FloatTensor([18198/4789, 18198/13409]),
          'anger':torch.cuda.FloatTensor([18198/3864, 18198/14334]),
          'surprise':torch.cuda.FloatTensor([18198/1845, 18198/16353]),
          'disgust':torch.cuda.FloatTensor([18198/3236, 18198/14962]),
          'fear':torch.cuda.FloatTensor([18198/1507, 18198/16691])}

def label_processing(l):
    s2, s7, happy, sad, anger, surprise, disgust, fear = 0,0,0,0,0,0,0,0
    if -2 <= l[0] and l[0] < -1:
        s7 = 1
    if -1 <= l[0] and l[0] < 0:
        s7 = 2
    if 0 <= l[0] and l[0] <= 0:
        s7 = 3
    if 0 < l[0] and l[0] <= 1:
        s7 = 4
    if 1 < l[0] and l[0] <= 2:
        s7 = 5
    if l[0] > 2:
        s7 = 6
    if l[0] >= 0:
        s2 = 1
    if l[1] > 0:
        happy = 1
    if l[2] > 0:
        sad = 1
    if l[3] > 0:
        anger = 1
    if l[4] > 0:
        surprise = 1
    if l[5] > 0:
        disgust = 1
    if l[6] > 0:
        fear = 1
    return s2, s7, happy, sad, anger, surprise, disgust, fear

def data_loader(data_set, name_list, batch_size, l_len, v_len, a_len, e):
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
            s2, s7, happy, sad, anger, surprise, disgust, fear = label_processing(label)
            label = {'sentiment_7':s7,'sentiment_2':s2,'happiness':happy,'sadness':sad,'anger':anger,'surprise':surprise,'disgust':disgust,'fear':fear}
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
            batch.append((l, v, a, l_mask, v_mask, a_mask, label[e]))
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

class Class_2(nn.Module):
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
        self.classifier = nn.Linear(dim, 2)
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

class Class_7(nn.Module):
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
        self.classifier = nn.Linear(dim, 7)
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
def train(model, iterator, weight, optimizer):
    model.train()
    epoch_loss, count = 0, 0
    iter_bar = tqdm(iterator, desc='Training')
    for _, batch in enumerate(iter_bar):
        count += 1
        optimizer.zero_grad()
        linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = zip(*batch)
        linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(label)
        logits_clsf = model(linguistic, visual, acoustic, l_mask, v_mask, a_mask)
        loss = nn.CrossEntropyLoss(weight=weight)(logits_clsf, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)  #梯度裁剪
        optimizer.step()
        iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
        epoch_loss += loss.item()
    return epoch_loss / count

def valid(model, iterator, weight):
    model.eval()
    epoch_loss, count = 0, 0
    with torch.no_grad():
        iter_bar = tqdm(iterator, desc='Validation')
        for _, batch in enumerate(iter_bar):
            count += 1
            linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = zip(*batch)
            linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(label)
            logits_clsf = model(linguistic, visual, acoustic, l_mask, v_mask, a_mask)
            loss = nn.CrossEntropyLoss(weight=weight, reduction='mean')(logits_clsf, label)
            iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss, count, epoch_loss / count

def test(model, iterator, e):
    model.eval()
    label_list, pred_list = [], []
    with torch.no_grad():
        iter_bar = tqdm(iterator, desc='Testing')
        for _, batch in enumerate(iter_bar):
            linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = zip(*batch)
            linguistic, visual, acoustic, l_mask, v_mask, a_mask, label = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.FloatTensor(l_mask), torch.cuda.FloatTensor(v_mask), torch.cuda.FloatTensor(a_mask), torch.cuda.LongTensor(label)
            pred = (model(linguistic, visual, acoustic, l_mask, v_mask, a_mask)).cpu().detach()
            pred = np.argsort(nn.LogSoftmax(dim=-1)(pred))[:,-1]
            label = label.cpu().detach()
            for i in range(len(label)):
                label_list.append(int(label[i]))
                pred_list.append(int(pred[i]))
    pred_acc = accuracy_score(label_list, pred_list)
    pred_f1 = f1_score(label_list, pred_list, average='weighted')
    return pred_acc, pred_f1

def run(model, data_set, weight_dict, train_list, valid_list, batch_size, learning_rate, epochs, e, name):
    log_file = log_dir+e+name+'.txt'
    with open(log_file, 'w') as log_f:
        log_f.write('epoch, train_loss, valid_loss\n')
    writer = SummaryWriter(log_dir)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(epochs):
        print('Epoch: ' + str(epoch+1))
        train_iterator = data_loader(data_set, train_list, batch_size, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, e=e)
        valid_iterator = data_loader(data_set, valid_list, batch_size, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, e=e)
        train_loss = train(model, train_iterator, weight_dict[e], optimizer)
        _, _, valid_loss = valid(model, valid_iterator, weight_dict[e])
        writer.add_scalars(e+name, {'train_loss':train_loss, 'valid_loss':valid_loss}, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss) 
        with open(log_file, 'a') as log_f:
            log_f.write('\n{epoch},{train_loss: 2.2f},{valid_loss: 2.2f}\n'.format(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss))
        if valid_loss == min(loss_list):
            stop = 0
            torch.save(model.state_dict(), os.path.join(save_dir, e+name+'_'+str(valid_loss)[:4]+'.pt'))
        else:
            stop += 1
            if stop >= 4:
                break
    writer.close()

def classify(model, name_list, e, shuffle=False):
    if shuffle:
        random.shuffle(name_list)
    valid_list = name_list[:int(len(name_list)*0.2)]
    train_list = name_list[int(len(name_list)*0.2):]
    run(model, data_set, weight, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, e=e, name='_model_1')
    
    valid_list = name_list[int(len(name_list)*0.2):int(len(name_list)*0.4)]
    train_list = name_list[:int(len(name_list)*0.2)] + name_list[int(len(name_list)*0.4):]
    run(model, data_set, weight, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, e=e, name='_model_2')
    
    valid_list = name_list[int(len(name_list)*0.4):int(len(name_list)*0.6)]
    train_list = name_list[:int(len(name_list)*0.4)] + name_list[int(len(name_list)*0.6):]
    run(model, data_set, weight, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, e=e, name='_model_3')
    
    valid_list = name_list[int(len(name_list)*0.6):int(len(name_list)*0.8)]
    train_list = name_list[:int(len(name_list)*0.6)] + name_list[int(len(name_list)*0.8):]
    run(model, data_set, weight, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, e=e, name='_model_4')
    
    valid_list = name_list[int(len(name_list)*0.8):]
    train_list = name_list[:int(len(name_list)*0.8)]
    run(model, data_set, weight, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, e=e, name='_model_5')

model = Class_2(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
print(get_parameter_number(model))  # 11126786
classify(model, name_list, e='sentiment_2')
classify(model, name_list, e='happiness')
classify(model, name_list, e='sadness')
classify(model, name_list, e='anger')
classify(model, name_list, e='surprise')
classify(model, name_list, e='disgust')
classify(model, name_list, e='fear')

model = Class_7(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, eps=EPS, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN).to(device)
classify(model, name_list, e='sentiment_7')
