import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# parameters
user = 'dango'

# data
l = torch.randn(128, 6, 3)
v = torch.randn(128, 7, 4)
a = torch.randn(128, 8, 5)
gt = []
for i in range(128):
    gt.append(random.randint(0, 5))
gt = torch.as_tensor(gt)
# print(l)
# print(v)
# print(a)
# print(gt)

def data_loader(linguistic, visual, acoustic, ground_truth, batch_size=2):
    count = 0
    while count < len(linguistic):
        batch = []
        if batch_size < len(linguistic) - count:
            size = batch_size
        else: 
            size = len(linguistic) - count
        batch.append((linguistic[count: count+size], visual[count: count+size], acoustic[count: count+size], ground_truth[count: count+size]))
        count += size
        yield batch

dl = data_loader(l, v, a, gt)
# print(next(dl))

# model
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
        return self.position_embeddings(position_ids)

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
            scores -= 10000.0 * (1.0 - mask.float())
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
            return self.fully_connected_2(xxx(self.fully_connected_1(x)))
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
        q += self.fully_connected[layer_num](self.self_attention[layer_num](q, k, v, mask))
        q += self.feed_forward[layer_num](self.normalization[layer_num*4+3](q))
        return q

class Model_1(nn.Module):
    def __init__(self, unify_dimension='Conv1D', position='True', activation='gelu', pooling='mean_1', l_dim, v_dim, a_dim, dim, l_len, v_len, a_len, eps, n_heads, n_layers, ffn):
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
        if pooling=='avg_1':
            self.fully_connected = nn.Linear(dim, dim)
        elif pooling=='avg_2':
            ?
        elif pooling=='max_1':
            ?
        elif pooling=='max_2':
            ?
        elif pooling=='avg+max_1':
            ?
        elif pooling=='avg+max_2':
            ?
        elif pooling=='flatten':
            ?
        self.classifier = nn.Linear(dim, 6)
    def forward(self, l, v, a, l_mask, v_mask, a_mask):
        l, v, a = self.unify_dimension(l, v, a)
        if self.position:
            l += self.linguistic_position(l)
            v += self.linguistic_position(v)
            a += self.linguistic_position(a)
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
        elif self.pooling=='avg+max_1':
            x = torch.mean(x, 1) + torch.max(x, 1)[0]
        elif self.pooling=='avg+max_2':
            x = torch.mean(x, 2) + torch.max(x, 2)[0]
        elif self.pooling=='flatten':
            x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

print(get_parameter_number(Embeddings(max_len=64)))

# run
def train(model, iterator, optimizer):
    model.train()
    epoch_loss, count = 0, 0
#     iter_bar = tqdm(iterator, desc='Training')
#     for _, batch in enumerate(iter_bar):
    for _, batch in enumerate(iterator):
        count += 1
        optimizer.zero_grad()
        linguistic, visual, acoustic, ground_truth = zip(*batch)
        logits_clsf = model(linguistic[0].cuda(), visual[0].cuda(), acoustic[0].cuda())
        loss = nn.CrossEntropyLoss()(logits_clsf, ground_truth[0].long().cuda())
        loss = loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #梯度裁剪
        optimizer.step()
#         iter_bar.set_description('Iter (loss=%4.4f)'%loss.item())
        epoch_loss += loss.item()
    return epoch_loss / count

def valid(model, iterator):
    model.eval()
    epoch_loss, count = 0, 0
    with torch.no_grad():
#         iter_bar = tqdm(iterator, desc='Validation')
#         for _, batch in enumerate(iter_bar):
        for _, batch in enumerate(iterator):
            count += 1
            linguistic, visual, acoustic, ground_truth = zip(*batch)
            logits_clsf = model(linguistic[0].cuda(), visual[0].cuda(), acoustic[0].cuda())
            loss = nn.CrossEntropyLoss()(logits_clsf, ground_truth[0].long().cuda())
            loss = loss.mean()
#             iter_bar.set_description('Iter (loss=%4.4f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss / count

def train_model(model, linguistic, visual, acoustic, ground_truth, batch_size=2, learning_rate=0.01, epochs=200):
    writer = SummaryWriter('/home/dango/multimodal/log/')
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(epochs):
        train_iterator = data_loader(linguistic, visual, acoustic, ground_truth, batch_size)
        valid_iterator = data_loader(linguistic, visual, acoustic, ground_truth, batch_size)
#         print('Epoch: ' + str(epoch+1))
        train_loss = train(model, train_iterator, optimizer)
        valid_loss = valid(model, valid_iterator)
        writer.add_scalar('Loss_value', valid_loss, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss) 
        if valid_loss == min(loss_list):
            stop = 0
            print(epoch+1, valid_loss)
        else:
            stop += 1
            if stop > 5:
                break
    return min(loss_list)

cls_loss = train_model(fenlei_model().to(device), l, v, a, gt)
