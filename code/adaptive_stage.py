import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm 
import warnings
import pandas as pd
from utils import simdatset,reproducibility,PreNorm,FeedForward

device = torch.device("cuda:0")
warnings.filterwarnings("ignore")


########  This process called adaptive stage, which cited from TAPE function (https://www.nature.com/articles/s41467-022-34550-9)


##########################
####    Prediction
##########################
def predict(test_x_1, peakname, celltypes, samplename,
            model_name=None, model=None, 
            adaptive=True, mode='Fast'):    
    if adaptive is True:
        if mode == 'high-resolution':
            Accessible_matrices_list = np.zeros((test_x_1.shape[0], len(celltypes), len(peakname)))
            Accessible_matrices_list.shape
            Pre_por = np.zeros((test_x_1.shape[0], len(celltypes)))
            print('Start adaptive training at high-resolution')
            model = torch.load(model_name + ".pth")
            for i in tqdm(range(len(test_x_1))):
                x = test_x_1[i:i+1, :, :]
                x.shape
                decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
                encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'mlp_head' in n]}]
                optimizerD = torch.optim.Adam(decoder_parameters, lr=1e-4)
                optimizerE = torch.optim.Adam(encoder_parameters, lr=1e-4)
                accessible_mcatrix, loss, test_pred = adaptive_stage(model, x, optimizerD,optimizerE) # steps=10, max_iter=50)
                accessible_mcatrix.shape
                test_pred.shape
                Accessible_matrices_list[i, :, :] = accessible_mcatrix
                Pre_por[i,:] = test_pred
                torch.cuda.empty_cache()
            accessible_mcatrix = pd.DataFrame(accessible_mcatrix,columns=peakname,index=celltypes)
            Pre_por = pd.DataFrame(Pre_por,columns=celltypes,index=samplename)
            CellTypeSigm = {}
            for i in range(len(celltypes)):
                cell_name = celltypes[i]
                sigm = Accessible_matrices_list[:,i,:]
                sigm = pd.DataFrame(sigm,columns=peakname,index=samplename)
                CellTypeSigm[cell_name] = sigm
            print('Adaptive stage is done')
            return accessible_mcatrix,CellTypeSigm, Pre_por
        elif mode == 'Fast':
            model = torch.load(model_name + ".pth")
            decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
            encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'mlp_head' in n]}]
            optimizerD = torch.optim.Adam(decoder_parameters, lr=1e-4)
            optimizerE = torch.optim.Adam(encoder_parameters, lr=1e-4)
            print('Start adaptive training for all the samples')            
            accessible_mcatrix, loss, Pre_por = adaptive_stage(model, x, optimizerD,optimizerE)
            accessible_mcatrix.shape
            Pre_por.shape
            print('Adaptive stage is done')
            accessible_mcatrix = pd.DataFrame(accessible_mcatrix,columns=peakname,index=celltypes)
            Pre_por = pd.DataFrame(Pre_por,columns=celltypes,index=samplename)
            return accessible_mcatrix, Pre_por
    else:
        if model_name is not None and model is None:
            model = torch.load(model_name+".pth")
        elif model is not None and model_name is None:
            model = model
        print('Predict cell type proportions without adaptive training')
        model.eval()
        model.state = 'test'
        data = torch.from_numpy(test_x_1).float().to(device)
        _, pred, _ = model(data)
        pred = pred.cpu().detach().numpy()
        pred = pd.DataFrame(pred, columns=celltypes, index=samplename)
        print('Prediction done')
        return pred

def adaptive_stage(model, data,optimizerD,optimizerE):
    data = torch.from_numpy(data).float().to(device)
    loss = []
    model.eval()
    model.state = 'test'
    _, ori_pred, ori_sigm = model(data)
    ori_sigm = ori_sigm.detach()
    ori_pred = ori_pred.detach()
    model.state = 'train' 
    for k in range(2):
        model.train()
        for i in range(2):
            reproducibility(seed=0)
            optimizerD.zero_grad()
            x_recon, _, sigm = model(data)
            batch_loss = F.mse_loss(x_recon, data)+F.mse_loss(sigm,ori_sigm)  ####  mse_loss
            batch_loss.backward()
            optimizerD.step()
            loss.append(F.mse_loss(x_recon, data).cpu().detach().numpy())   ####  mse_loss
        for i in range(2):
            reproducibility(seed=0)
            optimizerE.zero_grad()
            x_recon, pred, _ = model(data)
            # print(f'x_recon dim is:{x_recon.shape}')
            batch_loss = F.mse_loss(ori_pred, pred)+F.mse_loss(x_recon, data)
            batch_loss.backward()
            optimizerE.step()
            loss.append(F.mse_loss(x_recon, data).cpu().detach().numpy())
    model.eval()
    model.state = 'test'
    _, pred, sigm = model(data)
    return sigm.cpu().detach().numpy(), loss, pred.detach().cpu().numpy()


##########################
####    Train process
##########################

def train_predict(train_x_1, peakname, celltypes, samplename,
            model_name=None, model=None, 
            adaptive=True, mode='high-resolution'):  
    if adaptive is True:
        if mode == 'high-resolution':
            TestSigmList = np.zeros((train_x_1.shape[0], len(celltypes), len(peakname)))
            TestSigmList.shape
            TestPred = np.zeros((train_x_1.shape[0], len(celltypes)))
            print('Start adaptive training at high-resolution')
            model = torch.load(model_name + ".pth")
            for i in tqdm(range(len(train_x_1))):
                x = train_x_1[i:i+1, :, :]
                x.shape
                test_sigm, loss, test_pred = train_adaptive_stage(model, x) # steps=10, max_iter=50)
                test_sigm.shape
                test_pred.shape
                TestSigmList[i, :, :] = test_sigm
                TestPred[i,:] = test_pred
                torch.cuda.empty_cache()
            test_sigm = pd.DataFrame(test_sigm,columns=peakname,index=celltypes)
            TestPred = pd.DataFrame(TestPred,columns=celltypes,index=samplename)
            CellTypeSigm = {}
            for i in range(len(celltypes)):
                cellname = celltypes[i]
                sigm = TestSigmList[:,i,:]
                sigm = pd.DataFrame(sigm,columns=peakname,index=samplename)
                CellTypeSigm[cellname] = sigm
            print('Adaptive stage is done')
            return test_sigm,CellTypeSigm, TestPred


def train_adaptive_stage(model, data):
    data = torch.from_numpy(data).float().to(device)
    loss = []
    model.eval()
    model.state = 'test'
    _, pred, sigm = model(data)
    return sigm.cpu().detach().numpy(), loss, pred.detach().cpu().numpy()

