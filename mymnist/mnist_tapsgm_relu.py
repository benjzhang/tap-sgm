# %% [markdown]
# Simple implementation of MNIST with TAP-SGM with simple UNET

# %%
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 





inputchannels = 1
inputheight = 28
dimx = inputchannels * inputheight ** 2

depth = 1
hidden_units = 128



def construct_score_model(depth,hidden_units):
    chain = []
    chain.append(nn.Linear(int(dimx)+1,int(hidden_units),bias =True))
    chain.append(nn.GELU())

    for ii in range(depth-1):
        chain.append(nn.Linear(int(hidden_units),int(hidden_units),bias = True))
        chain.append(nn.GELU())
    chain.append(nn.Linear(int(hidden_units),dimx,bias = True))    

    return nn.Sequential(*chain)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scorenet =  construct_score_model(depth,hidden_units)
scorenet = scorenet.to(device)

loaded = torch.load('pushforward_samples')
loaded = loaded.to(device)
# %%
# Train the scorenet
def calc_loss(score_network: torch.nn.Module, x: torch.Tensor,Tmin,Tmax,eps) -> torch.Tensor:
    # x: (batch_size, nch) is the training data
    
    # sample the time
    t = torch.rand([x.size(0) ]).to(x) * (Tmax-Tmin+eps) #+ [1 for _ in range(x.ndim - 1)]).to(x) * (Tmax-Tmin+eps)

    # calculate the terms for the posterior log distribution

    sigmas = torch.sqrt(1 - torch.exp(-t))
    sigmas = sigmas.view(x.shape[0],*([1]*len(x.shape[1:])))
    noise = torch.randn_like(x) * sigmas

    tenlarge = t.repeat(784,1).T
    perturbed_samples = x*torch.exp(-0.5 * tenlarge) + noise

    target = - 1/(sigmas ** 2) * noise
    score_eval_samples = torch.cat((t.reshape(-1,1),perturbed_samples),1)

    scores = -perturbed_samples + score_network(score_eval_samples)

    target = target.view(target.shape[0],-1)
    scores = scores.view(scores.shape[0],-1)
    loss = 0.5 * ((scores-target)**2).sum(dim = -1)

    return loss.mean(dim = 0)


    # int_beta = (0.1 + 0.5 * (20 - 0.1) * t) * t  # integral of beta
    # mu_t = x * torch.exp(-0.5 * int_beta)
    # var_t = -torch.expm1(-int_beta)
    # x_t = torch.randn_like(x) * var_t ** 0.5 + mu_t
    # grad_log_p = -(x_t - mu_t) / var_t  # (batch_size, nch)

    # # calculate the score function
    # score = score_network(x_t, t)  # score: (batch_size, nch)

    # # calculate the loss function
    # loss = (score - grad_log_p) ** 2
    # lmbda_t = var_t
    # weighted_loss = lmbda_t * loss
    # return torch.mean(weighted_loss)

# %%
## training

loaded = torch.load('pushforward_samples')
loaded = loaded.to(device)

scorenet.train()
opt = torch.optim.Adam(scorenet.parameters(),lr = 0.0005)

epochs = 100000
for step in range(epochs):

    opt.zero_grad()
    randind = torch.randint(0,59999,(256,))
    data = torch.tensor(loaded[randind,:,:,:])
    data = data.reshape(data.shape[0],-1).to(device)
    # training step
    loss = calc_loss(scorenet, data,0,5,1e-4)
    loss.backward()
    opt.step()
    if not step%100:
        print(loss,step)

scorenet.eval()
torch.save(scorenet,'mnist_scorenet_tapsgm_ffjord_ReLU_T5_lr0005_iter100k_strongprior_shallow')
