import torch
import warnings
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F


device = torch.device("cuda:0")
warnings.filterwarnings("ignore")

def training_stage(model, train_loader, optimizer, epochs=128):   
    model.train()
    model.state = 'train'
    loss = []
    recon_loss = []
    accumulation_steps = 8  
    for i in tqdm(range(epochs)):
        for k, (data, label) in enumerate(train_loader):
            # reproducibility(seed=0)
            optimizer.zero_grad()
            x_recon, cell_prop, sigm = model(data)
            batch_loss = F.mse_loss(cell_prop, label.squeeze(dim=1)) + F.mse_loss(x_recon, data.squeeze(dim=1))   ###  F.l1_loss(x_recon, data)
            batch_loss /= accumulation_steps  
            batch_loss.backward()
            if (k + 1) % accumulation_steps == 0:
                optimizer.step()
            loss.append(F.mse_loss(cell_prop, label).cpu().detach().numpy())
            recon_loss.append(F.mse_loss(x_recon, data).cpu().detach().numpy())        
    return model, loss, recon_loss

def showloss(loss):
    sns.set(style='whitegrid', palette='husl')  # Customize the seaborn style and color palette
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(loss, color='#b2bec3', linestyle='-', linewidth=0.8, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over iterations')
    plt.legend()  # Show legend for the plotted line
    plt.grid(False)  # Remove grid lines
    plt.show()