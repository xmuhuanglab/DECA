#############################################################    Test function    #############################################################
import time
import torch
import pickle
import anndata
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device("cuda:0")
warnings.filterwarnings("ignore")

def test_generate_simulated_data(test_in,outname=None):

    
    ##################
    #### load file
    ##################
    print('Reading Bulk ATAC dataset, this may take some minutes')
    
    sc_data=pd.read_csv(test_in)
    colnames=list(sc_data.columns)
    sc_data.index=sc_data[colnames[0]]
    peakname=sc_data[colnames[0]]
    sc_data=sc_data.drop(columns=[colnames[0]])
    simudata = anndata.AnnData(X=np.array(sc_data.T),
                               var=pd.DataFrame(index=peakname))

    return simudata,peakname


def ProcessInputData(test_data, test_name,test_path, sep=None, datatype='counts', variance_threshold=0.95,
                     scaler="mms",
                     genelenfile=None):


    if type(test_data) is anndata.AnnData:
        pass
    elif type(test_data) is str:
        test_data = anndata.read_h5ad(test_data)
    test_data=test_data
    test_x = pd.DataFrame(test_data.X, columns=test_data.var.index)
    test_y = test_data.obs
    
    test_x=test_x

    samplename = test_x.index

    ### MinMax process
    print('Scaling...')
    test_x = np.log(test_x + 1)

    colors = sns.color_palette('RdYlBu', 10)
    fig = plt.figure(figsize=(3.5, 3))
    sns.histplot(data=np.mean(test_x, axis=0), kde=True, color=colors[7],edgecolor=None)
    plt.legend(title='datatype', labels=[ 'testdata'])

    plt.show()

    if scaler=='ss':
        print("Using standard scaler...")
        ss = StandardScaler()
        ss_test_x = ss.fit_transform(test_x.T).T
        fig = plt.figure(figsize=(3.5, 3))
        sns.histplot(data=np.mean(ss_test_x, axis=0), kde=True, color=colors[7],edgecolor=None)
        plt.legend(title='datatype', labels=[ 'testdata'])

        plt.show()

        return  ss_test_x, samplename

    elif scaler == 'mms':
        print("Using min-max scale...")
        mms = MinMaxScaler()
        mms_test_x = mms.fit_transform(test_x.T).T
        fig = plt.figure(figsize=(3.5, 3))
        
        sns.histplot(data=np.mean(mms_test_x, axis=0), kde=True, color=colors[7],edgecolor=None)
        plt.legend(title='datatype', labels=['testdata'])

        plt.show()

        return  mms_test_x, samplename


torch.backends.cudnn.benchmark = True

def reproducibility(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential( 
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class simdatset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float().to(device)
        y = torch.from_numpy(self.Y[index]).float().to(device)
        return x, y


def train_model(train_x, train_y,
                model_name=None,
                batch_size=100, epochs=100):
    
    train_loader = DataLoader(simdatset(train_x_1, train_y_1), batch_size=batch_size, shuffle=True)
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
    
    print('Training is done')
    print('prediction loss is:')
    showloss(loss)
    print('reconstruction loss is:')
    showloss(reconloss)
    if model_name is not None:
        print('Model is saved')
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
    



def training_stage(model, train_loader, optimizer, epochs=128): 
    
    model.train()
    model.state = 'train'
    loss = []
    recon_loss = []
    accumulation_steps = 8  # 累积4个批次的梯度
    for i in tqdm(range(epochs)):
        for k, (data, label) in enumerate(train_loader):
            # reproducibility(seed=0)
            optimizer.zero_grad()
            x_recon, cell_prop, sigm = model(data)

            batch_loss = F.mse_loss(cell_prop, label.squeeze(dim=1)) + F.mse_loss(x_recon, data.squeeze(dim=1))   ###  F.l1_loss(x_recon, data)
            batch_loss /= accumulation_steps  # 将损失除以累积步骤数
            batch_loss.backward()
            if (k + 1) % accumulation_steps == 0:
                optimizer.step()
            loss.append(F.mse_loss(cell_prop, label).cpu().detach().numpy())
            recon_loss.append(F.mse_loss(x_recon, data).cpu().detach().numpy())
            
    return model, loss, recon_loss

def showloss(loss):
    sns.set(style='whitegrid', palette='husl')  # Customize the seaborn style and color palette
    plt.figure(figsize=(3.5, 3))
    plt.plot(loss, color='#34495e', linestyle='-', label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Over Iterations')
    plt.legend()  # Show legend for the plotted line
    plt.grid(False)  # Remove grid lines
    plt.show()


def predict(test_x_1, genename, celltypes, samplename,
            model_name=None, model=None, 
            adaptive=True, mode='high-resolution'):
    
    if adaptive is True:
        if mode == 'high-resolution':
            TestSigmList = np.zeros((test_x_1.shape[0], len(celltypes), len(genename)))
            TestSigmList.shape
            TestPred = np.zeros((test_x_1.shape[0], len(celltypes)))
            print('Start adaptive training at high-resolution')
            model = torch.load(model_name + ".pth")

            for i in tqdm(range(len(test_x_1))):
                x = test_x_1[i:i+1, :, :]
                x.shape
                decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
                encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'mlp_head' in n]}]
                optimizerD = torch.optim.Adam(decoder_parameters, lr=1e-4)
                optimizerE = torch.optim.Adam(encoder_parameters, lr=1e-4)
                test_sigm, loss, test_pred = predict_adaptive_stage(model, x, optimizerD,optimizerE) # steps=10, max_iter=50)
                test_sigm.shape
                test_pred.shape
                TestSigmList[i, :, :] = test_sigm
                TestPred[i,:] = test_pred
                torch.cuda.empty_cache()
            test_sigm = pd.DataFrame(test_sigm,columns=genename,index=celltypes)
            TestPred = pd.DataFrame(TestPred,columns=celltypes,index=samplename)
            CellTypeSigm = {}
            for i in range(len(celltypes)):
                cellname = celltypes[i]
                sigm = TestSigmList[:,i,:]
                sigm = pd.DataFrame(sigm,columns=genename,index=samplename)
                CellTypeSigm[cellname] = sigm
            print('Adaptive stage is done')

            return test_sigm,CellTypeSigm, TestPred

        elif mode == 'overall':
            model = torch.load(model_name + ".pth")
            decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
            encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'mlp_head' in n]}]
            optimizerD = torch.optim.Adam(decoder_parameters, lr=1e-4)
            optimizerE = torch.optim.Adam(encoder_parameters, lr=1e-4)
            print('Start adaptive training for all the samples')
            
            test_sigm, loss, test_pred = predict_adaptive_stage(model, x, optimizerD,optimizerE)
            test_sigm.shape
            test_pred.shape
            print('Adaptive stage is done')
            test_sigm = pd.DataFrame(test_sigm,columns=genename,index=celltypes)
            test_pred = pd.DataFrame(test_pred,columns=celltypes,index=samplename)

            return test_sigm, test_pred

    else:
        if model_name is not None and model is None:
            model = torch.load(model_name+".pth")
        elif model is not None and model_name is None:
            model = model
        print('Predict cell fractions without adaptive training')
        model.eval()
        model.state = 'test'
        data = torch.from_numpy(test_x_1).float().to(device)
        _, pred, _ = model(data)
        pred = pred.cpu().detach().numpy()
        pred = pd.DataFrame(pred, columns=celltypes, index=samplename)
        print('Prediction is done')
        return pred

def predict_adaptive_stage(model, data,optimizerD,optimizerE):
    data = torch.from_numpy(data).float().to(device)
    loss = []
    model.eval()
    model.state = 'test'
    _, ori_pred, ori_sigm = model(data)
    ori_sigm = ori_sigm.detach()
    ori_pred = ori_pred.detach()

    return ori_sigm.cpu().detach().numpy(), loss, ori_pred.detach().cpu().numpy()