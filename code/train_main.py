import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from utils import simdatset,reproducibility,PreNorm,FeedForward
from train_stage import training_stage,showloss

import warnings
from tqdm import tqdm 

device = torch.device("cuda:0")
warnings.filterwarnings("ignore")

def train_model(train_x, train_y,
                model_name=None,
                batch_size=100, input_size=1,class_size=4,output_dim=1,state='train',epochs=100):   
    train_loader = DataLoader(simdatset(train_x, train_y), batch_size=batch_size, shuffle=True)
    model = ViT(
        seq_len=input_size,
        patch_size=100,
        num_classes=class_size,
        output_dim=output_dim,
        dim=256,
        depth=9, ## 6
        heads=8, ## 8
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
        state=state,
        batch_size=batch_size
    ).to(device)
    # reproducibility(seed=0)
    optimizer = Adam(model.parameters(), lr=1e-4)
    print('Start training')
    model, loss, reconloss = training_stage(model, train_loader, optimizer, epochs=epochs) 
    print('Training done')
    print('Prediction loss is:')
    showloss(loss)
    print('Reconstruction accessibility matrix loss is:')
    showloss(reconloss)
    if model_name is not None:
        print('Model saved')
        torch.save(model, model_name+".pth")
    return model


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes,output_dim,batch_size,
                 dim, depth, heads, mlp_dim,state, channels = 1,
                 dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0
        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.decoder = nn.Sequential(nn.Linear(num_classes, 64, bias=False),
                                     nn.Linear(64, 128, bias=False),
                                     nn.Linear(128, 256, bias=False),
                                     nn.Linear(256, 512, bias=False),
                                     nn.Linear(512, seq_len, bias=False))   
    def refraction(self,x):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return x/x_sum
    def decode(self, z):
        return self.decoder(z)
    def sigmatrix(self):
        w0 = (self.decoder[0].weight.T)
        w1 = (self.decoder[1].weight.T)
        w2 = (self.decoder[2].weight.T)
        w3 = (self.decoder[3].weight.T)
        w4 = (self.decoder[4].weight.T)
        w01 = (torch.mm(w0, w1))
        w02 = (torch.mm(w01, w2))
        w03 = (torch.mm(w02, w3))
        w04 = (torch.mm(w03, w4))
        return F.relu(w04)
    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        sigmatrix = self.sigmatrix()
        cls_tokens, _ = unpack(x, ps, 'b * d')
        logits=self.mlp_head(cls_tokens)
        # z=logits       
        # if self.state == 'train':
        #     pass
        # elif self.state == 'test':
        #     z = F.relu(z)
        #     z = self.refraction(z)       
        # x_recon = torch.mm(z, sigmatrix)
        class_probs = torch.sigmoid(logits)
        x_recon = torch.mm(class_probs, sigmatrix)       
        return  x_recon, class_probs, sigmatrix