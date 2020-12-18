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
log_dir = '/home/'+user+'/par_log/log_crf_8/'
L_DIM = 300
V_DIM = 35
A_DIM = 74
L_LEN = 30
V_LEN = 40
A_LEN = 50
LR = 0.0001
CLIP = 1.0
DROP = 0.0
EPOCHS = 99
BATCH = 64
DIM = 96
N_HEADS = 6
FFN = 2
N_LAYERS = 1
ALPHA = 0.9
trans_matrix = torch.tensor(
 [[0.704, 0.079, 0.035, 0.072, 0.041, 0.038],
 [0.127, 0.493, 0.13,  0.073, 0.09,  0.08 ],
 [0.084, 0.172, 0.428, 0.097, 0.153, 0.081],
 [0.279, 0.12,  0.12,  0.36,  0.062, 0.092],
 [0.068, 0.124, 0.206, 0.072, 0.447, 0.092],
 [0.074, 0.176, 0.12,  0.132, 0.124, 0.408]])
P_LEN = 8

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

def masking(m, m_len):
    if len(m) >= m_len:
        m_mask = np.ones(m_len)
    else:
        m_mask = np.concatenate((np.ones(len(m)), np.zeros(m_len - len(m))))
    m = np.concatenate([m, np.zeros([m_len]+list(m.shape[1:]))],axis=0)[:m_len,...]
    for i in range(len(m)):
        for j in range(len(m[i])):
            if math.isinf(m[i][j]) or math.isnan(m[i][j]):
                m[i][j] = -71.
    return m, m_mask

def label_processing(l):
    label = l[1:]
    label[0] = 1 if label[0] > 0 else 0
    label[1] = 1 if label[1] > 0 else 0
    label[2] = 1 if label[2] > 0 else 0
    label[3] = 1 if label[3] > 0 else 0
    label[4] = 1 if label[4] > 0 else 0
    label[5] = 1 if label[5] > 0 else 0
    return label

def data_loader(data_set, name_list, batch_size):
    random.shuffle(name_list)
    count = 0
    while count < len(name_list):
        batch = []
        size = min(batch_size, len(name_list) - count)
        for _ in range(size):
            l_temp, v_temp, a_temp, label_temp, mask_temp = [], [], [], [], []
            for i in range(P_LEN):
                if name_list[count][i] != 'no_name':
                    l, l_mask = masking(data_set.computational_sequences['linguistic'].data[name_list[count][i]]["features"][-L_LEN:], L_LEN)
                    v, v_mask = masking(data_set.computational_sequences['visual'].data[name_list[count][i]]["features"][-V_LEN:], V_LEN)
                    a, a_mask = masking(data_set.computational_sequences['acoustic'].data[name_list[count][i]]["features"][-A_LEN:], A_LEN)
                    label = label_processing(data_set.computational_sequences['label'].data[name_list[count][i]]["features"][0])
                    mask = 1
                else:
                    l, v, a = np.zeros((L_LEN, L_DIM)), np.zeros((V_LEN, V_DIM)), np.zeros((A_LEN, A_DIM))
                    l_mask, v_mask, a_mask = np.zeros(L_LEN), np.zeros(V_LEN), np.zeros(A_LEN)
                    label = np.array([0, 0, 0, 0, 0, 0])
                    mask = 0
                l_temp.append(np.expand_dims(l, axis=0))
                v_temp.append(np.expand_dims(v, axis=0))
                a_temp.append(np.expand_dims(a, axis=0))
                label_temp.append(np.expand_dims(label, axis=0))
                mask_temp.append(mask)
            batch.append((np.concatenate(l_temp, axis=0), np.concatenate(v_temp, axis=0), np.concatenate(a_temp, axis=0), np.concatenate(label_temp, axis=0), np.asarray(mask_temp)))
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
    def __init__(self, l_dim, v_dim, a_dim, dim, l_len, v_len, a_len, n_heads, n_layers, ffn):
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
        self.normalization = nn.LayerNorm(dim)
        self.drop = nn.Dropout(DROP)
        self.classifier = nn.Linear(dim, 6)
    def forward(self, l, v, a):  # (batch, x_len, x_dim)
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
        return self.classifier(x)  # (batch, 6)

class TransferNet(nn.Module):
    def __init__(self, trans_matrix, if_trans_fintuning = True, if_consider_neutral = False):
        super().__init__()
        init_alpha = torch.FloatTensor([ALPHA])
        self.alpha = nn.Parameter(init_alpha,requires_grad=True)
        self.transitions = nn.Parameter(trans_matrix, requires_grad=if_trans_fintuning)
    def log_sum_exp(self,vec, m_size):
        _, idx = torch.max(vec, 1)
        max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)
        return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)
    def forward(self, feats):
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_in_size = feats.size(-1)
        tag_out_size = self.transitions.size(-1)
        alpha_norm = torch.sigmoid(self.alpha)
        transitions_norm = torch.softmax(self.transitions,axis = -1)
        seq_iter = enumerate(feats.transpose(1, 0))
        p_e = []
        for idx, cur_values in seq_iter:
            if idx == 0:
                cur_emo_state = cur_values
            else:
                cur_emo_state = alpha_norm * cur_values + (1. - alpha_norm) * transfer_energy
            p_e.append(torch.unsqueeze(cur_emo_state,1))
            if idx == seq_len-1:
                continue
            cur_emo_state = cur_emo_state.contiguous().view(batch_size, tag_out_size,1).expand(batch_size, tag_out_size, tag_out_size)
            trans_to_next_ = cur_emo_state * transitions_norm.view(1, tag_out_size, tag_out_size).expand(batch_size, tag_out_size, tag_out_size)
            transfer_energy = self.log_sum_exp(trans_to_next_, tag_out_size)
        return torch.cat(p_e,dim =1)

class State_Transfer(nn.Module):
    def __init__(self, l_dim, v_dim, a_dim, dim, l_len, v_len, a_len, n_heads, n_layers, ffn, trans_matrix):
        super().__init__()
        self.feature = Multi_class(l_dim=l_dim, v_dim=v_dim, a_dim=a_dim, dim=dim, l_len=l_len, v_len=v_len, a_len=a_len, n_heads=n_heads, n_layers=n_layers, ffn=ffn)
        self.transfer = TransferNet(trans_matrix=torch.FloatTensor(trans_matrix), if_consider_neutral=False, if_trans_fintuning=True)
    def forward(self, l, v, a):  # (batch, y_len, x_len, x_dim)
        feats_list = []
        for i in range(l.shape[1]):
            temp_l, temp_v, temp_a = l[:,i,:,:], v[:,i,:,:], a[:,i,:,:]
            temp_feats = self.feature(temp_l, temp_v, temp_a)
            feats_list.append(temp_feats.unsqueeze(1))
        feats = torch.cat(feats_list, dim=1)
        return self.transfer(feats = feats)  # (batch, y_len, 6)

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
        linguistic, visual, acoustic, label, mask = zip(*batch)
        linguistic, visual, acoustic, label, mask = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.LongTensor(label), torch.cuda.LongTensor(mask)
        logits_clsf = model(linguistic, visual, acoustic)
        loss = multi_circle_loss(logits_clsf, label)
        loss = (loss*mask).mean()
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
            linguistic, visual, acoustic, label, mask = zip(*batch)
            linguistic, visual, acoustic, label, mask = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.LongTensor(label), torch.cuda.LongTensor(mask)
            logits_clsf = model(linguistic, visual, acoustic)
            loss = multi_circle_loss(logits_clsf, label)
            loss = (loss*mask).mean()
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
        train_iterator = data_loader(data_set, train_list, batch_size)
        valid_iterator = data_loader(data_set, valid_list, batch_size)
        train_loss = train(model, train_iterator, optimizer)
        _, _, valid_loss = valid(model, valid_iterator)
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
            if stop >= 4:
                break
    writer.close()

model_1 = State_Transfer(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN, trans_matrix=trans_matrix).to(device)
valid_list = train_name_list[:int(len(train_name_list)*0.2)]
train_list = train_name_list[int(len(train_name_list)*0.2):]
run(model_1, data_set, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_1')
model_2 = State_Transfer(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN, trans_matrix=trans_matrix).to(device)
valid_list = train_name_list[int(len(train_name_list)*0.2):int(len(train_name_list)*0.4)]
train_list = train_name_list[:int(len(train_name_list)*0.2)] + name_list[int(len(train_name_list)*0.4):]
run(model_2, data_set, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_2')

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
            linguistic, visual, acoustic, label, mask = zip(*batch)
            linguistic, visual, acoustic, label, mask = torch.cuda.FloatTensor(linguistic), torch.cuda.FloatTensor(visual), torch.cuda.FloatTensor(acoustic), torch.cuda.LongTensor(label), torch.cuda.LongTensor(mask)
            pred_1 = (model_1(linguistic, visual, acoustic)).cpu().detach()
            pred_2 = (model_2(linguistic, visual, acoustic)).cpu().detach()
            pred = pred_1 * 0.6 + pred_2 * 0.4
            zero = torch.zeros_like(pred)
            one = torch.ones_like(pred)
            pred = torch.where(pred > threshold, one, zero)
            label = label.cpu().detach()
            for i in range(len(mask)):
                for j in range(P_LEN):
                    if int(mask[i][j]) == 1:
                        label_happ.append(int(label[i][j][0]))
                        label_sadn.append(int(label[i][j][1]))
                        label_ange.append(int(label[i][j][2]))
                        label_surp.append(int(label[i][j][3]))
                        label_disg.append(int(label[i][j][4]))
                        label_fear.append(int(label[i][j][5]))
                        soft_happ.append(int(pred[i][j][0]))
                        soft_sadn.append(int(pred[i][j][1]))
                        soft_ange.append(int(pred[i][j][2]))
                        soft_surp.append(int(pred[i][j][3]))
                        soft_disg.append(int(pred[i][j][4]))
                        soft_fear.append(int(pred[i][j][5]))
                    else:
                        break
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

# log_crf_8
model_1 = State_Transfer(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN, trans_matrix=trans_matrix).to(device)
model_1.load_state_dict(torch.load(log_dir + '?.pt'))
model_2 = State_Transfer(l_dim=L_DIM, v_dim=V_DIM, a_dim=A_DIM, dim=DIM, l_len=L_LEN, v_len=V_LEN, a_len=A_LEN, n_heads=N_HEADS, n_layers=N_LAYERS, ffn=FFN, trans_matrix=trans_matrix).to(device)
model_2.load_state_dict(torch.load(log_dir + 'model_1_1.83.pt'))

test_iterator = data_loader(data_set, test_name_list, batch_size=BATCH)
th = 0
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
# balabce 1.0116603047357247
