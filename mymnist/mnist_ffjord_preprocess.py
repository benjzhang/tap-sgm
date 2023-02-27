# %% [markdown]
# Preprocess MNIST with FFJORD map

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

# %%
## FFJORD things
import lib.layers as layers
import lib.utils as utils
import lib.odenvp as odenvp
import lib.multiscale_parallel as multiscale_parallel

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log


# %%
def ffjord_create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    if args.multiscale:
        model = odenvp.ODENVP(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            nonlinearity=args.nonlinearity,
            alpha=args.alpha,
            cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns},
        )
    elif args.parallel:
        model = multiscale_parallel.MultiscaleParallelCNF(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            alpha=args.alpha,
            time_length=args.time_length,
        )
    else:
        if args.autoencode:

            def build_cnf():
                autoencoder_diffeq = layers.AutoencoderDiffEqNet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.AutoencoderODEfunc(
                    autoencoder_diffeq=autoencoder_diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                )
                return cnf
        else:

            def build_cnf():
                diffeq = layers.ODEnet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.ODEfunc(
                    diffeq=diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    train_T=args.train_T,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                )
                return cnf

        chain = [layers.LogitTransform(alpha=args.alpha)] if args.alpha > 0 else [layers.ZeroMeanTransform()]
        chain = chain + [build_cnf() for _ in range(args.num_blocks)]
        if args.batch_norm:
            chain.append(layers.MovingBatchNorm2d(data_shape[0]))
        model = layers.SequentialFlow(chain)
    return model

# %%
# import MNIST samples

transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.MNIST('mnist_data',train = True,download = True, transform = transform)


# %%
## Define scorenetwork 

inputchannels = 1
inputheight = 28


# %%
## Load preconditioning map

checkpoint = torch.load('ffjord_mnist_bad_checkpt.pth',map_location = torch.device('cpu'))
preconditionerargs = checkpoint['args']

data_shape = (inputchannels, inputheight, inputheight)

regularization_fns, regularization_coeffs = create_regularization_fns(preconditionerargs)
model = ffjord_create_model(preconditionerargs,data_shape,regularization_fns)
model.load_state_dict(checkpoint['state_dict'])

# %%
print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# %%
## Pushforward samples

# %%
loader = torch.utils.data.DataLoader(dataset,10)
blah = enumerate(loader)
a, (blah,b) = next(blah)

pushforward_z = torch.zeros(60000,1,28,28)

for ii in range(60):
    blah_sampled = blah[(ii*1000):((ii+1)*1000),:,:,:]
    pushforward_z[(ii*1000):((ii+1)*1000),:,:,:] = model(blah_sampled,reverse = False).view(-1,*data_shape)

torch.save(pushforward_z,'pushforward_samples')