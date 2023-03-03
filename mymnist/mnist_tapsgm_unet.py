# %% [markdown]
# Simple implementation of MNIST with TAP-SGM

# %%
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
from models.unet import UNet





inputchannels = 1
inputheight = 28
dimx = inputchannels * inputheight ** 2


scorenet = UNet(input_channels = inputchannels,
                input_height = inputheight,
                ch = 32,
                ch_mult = (1,2,2),
                num_res_blocks=2,
                attn_resolutions=(16,),
                resamp_with_conv=True,
                )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scorenet = scorenet.to(device)

loaded = torch.load('pushforward_samples')
loaded = loaded.to(device)
# %%
# Train the scorenet

def calc_loss(score_network: torch.nn.Module, x: torch.Tensor,Tmin,Tmax,eps) -> torch.Tensor:
    # x: (batch_size, nch) is the training data
    
    # sample the time
    t = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * (Tmax-Tmin+eps)

    # calculate the terms for the posterior log distribution

    sigmas = torch.sqrt(1 - torch.exp(-t))
    noise = torch.randn_like(x) * sigmas

    perturbed_samples = x*torch.exp(-0.5 * t) + noise

    target = - 1/(sigmas ** 2) * noise

    scores = -perturbed_samples + score_network(perturbed_samples,t.squeeze())

    target = target.view(target.shape[0],-1)
    scores = scores.view(scores.shape[0],-1)
    loss = 0.5 * ((scores-target)**2)

    return (loss.view(x.size(0), -1).sum(1, keepdim=False)).mean()

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
scorenet.train()
opt = torch.optim.Adam(scorenet.parameters(),lr = 0.01)

epochs = 100000
for step in range(epochs):

    opt.zero_grad()
    randind = torch.randint(0,59999,(64,))
    data = torch.tensor(loaded[randind,:,:,:])

    # training step
    loss = calc_loss(scorenet, data,0,5,1e-4)
    loss.backward()
    opt.step()
    print(loss)

scorenet.eval()
torch.save(scorenet,'mnist_scorenet_tapsgm_ffjord_T5_lr001_iter100k_strongprior')
