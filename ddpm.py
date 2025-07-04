"""DDPM module."""

import warnings

import numpy as np
import pandas as pd
# import torch
from torch import optim, nn, load
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
import random

import schedulefree

from copy import deepcopy

from modules import *
from gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from utils_train import get_model, update_ema
from data import prepare_fast_dataloader

from tqdm import tqdm
from base import BaseSynthesizer, random_state



class RandomWalkDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.size =  len(dataset)
    def __iter__(self):
        while True:
            yield self.dataset[random.randint(0,self.size-1)]
            
def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)


class DDPM(BaseSynthesizer):

    def __init__(self, lr,  layers, num_timesteps, model_name = "mlp", dim_t = 128, gaussian_loss_type = "mse", 
                 multinomial_loss_type = 'vb_all', parametrization = 'x0', scheduler = "cosine", is_y_cond = False, weight_decay = 0,
                batch_size=500, log_every=100,  verbose=False, epochs=300, device="cuda", save_as = "_dqn_", load_as = "_dqn_"):

        assert batch_size % 2 == 0
    
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_every = log_every
        self.disbalance = None
        self.steps = epochs
        self.layers = layers
        self.init_lr = lr
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.gaussian_loss_type = gaussian_loss_type
        self.multinomial_loss_type = multinomial_loss_type
        self.parametrization = parametrization
        self.scheduler = scheduler
        
        self.is_y_cond = is_y_cond
        self.dim_t = dim_t
        self.model_name = model_name
        
        self._transformer = None
        self._data_sampler = None
        

        self.ema_every = 1000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_as = save_as
        self.load_as = load_as
        self._verbose = verbose
        self.loss_values = None
        
    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss
    



    @random_state
    def fit(self, dataset):
        
        K = np.array(dataset.get_category_sizes('train'))
        if len(K) == 0:
            K = np.array([0])
        print("K", K)

        num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
        d_in = np.sum(K) + num_numerical_features
 
        print("features", d_in)
    
        model_params = {"d_in" : d_in,
                        "is_y_cond" : self.is_y_cond ,
                        "num_classes" : dataset.n_classes,
                        "rtdl_params" : {"d_layers" : [self.layers, self.layers],
                                         "dropout" : 0.0
                                        },
                        "dim_t" : self.dim_t
                       }

        print(model_params)

        mlp = get_model(self.model_name,
                        model_params,
                        num_numerical_features,
                        category_sizes=dataset.get_category_sizes('train'))
        
        mlp.to(self.device)    
        
        self.train_iter = prepare_fast_dataloader(dataset, split='train', batch_size = self.batch_size)

        self.diffusion = GaussianMultinomialDiffusion(num_classes=K,
                                                      num_numerical_features=num_numerical_features,
                                                      denoise_fn=mlp,
                                                      gaussian_loss_type=self.gaussian_loss_type,
                                                      num_timesteps=self.num_timesteps,
                                                      multinomial_loss_type=self.multinomial_loss_type,
                                                      parametrization=self.parametrization,
                                                      scheduler=self.scheduler,
                                                      device = self.device)
        self.diffusion.to(self.device)
        self.diffusion.train()

        print("diffusion ready")
        
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()
        
        # self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = schedulefree.AdamWScheduleFree(self.diffusion.parameters(), lr=self.lr)
        self.optimizer.train()
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 
                                        T_0 = 4000,# Number of iterations for the first restart
                                        T_mult = 1, # A factor increases TiTi after a restart
                                        eta_min = 1e-6) # Minimum learning rate

        step_iterator = tqdm(range(self.steps), disable=(not self._verbose))
        
        if self._verbose:
            description = 'mloss ({mloss:.2f}) | gloss ({gloss:.2f})'
            step_iterator.set_description(description.format(mloss=0, gloss=0))
        
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        mloss, gloss = 0, 0
        
        for step in step_iterator:

            
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)


            # LR UPDATE
            # self._anneal_lr(step)
            # scheduler.step()
            

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)


            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())
            

            if self._verbose:
                step_iterator.set_description(
                    description.format(mloss=mloss, gloss=gloss)
                )
        
        torch.save(self.diffusion._denoise_fn.state_dict(), "./SVG_model/model_" + self.save_as + "_diffusion.pt")
        torch.save(self.ema_model.state_dict(), "./SVG_model/model_" + self.save_as + "_ema.pt")

    @random_state
    def sample(self, dataset, num_samples = 0, batch_size = 2000):
        
        if num_samples == 0:
        
            num_samples, num_numerical_features = dataset.X_num["train"].shape
            
        else:
            
            num_numerical_features = dataset.X_num["train"].shape[1]
        
        self.batch_size = batch_size
        
        K = np.array(dataset.get_category_sizes('train'))
        if len(K) == 0:
            K = np.array([0])
        print(K)

        num_numerical_features_ = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
        d_in = np.sum(K) + num_numerical_features_
        
        
        print("features", d_in)
    
        model_params = {"d_in" : d_in,
                        "is_y_cond" : self.is_y_cond,
                        "num_classes" : dataset.n_classes,
                        "rtdl_params" : {"d_layers" : [self.layers, self.layers],
                                         "dropout" : 0.0
                                        },
                        "dim_t" : self.dim_t
                       }

        print(model_params)

        mlp = get_model(self.model_name,
                        model_params,
                        num_numerical_features_,
                        category_sizes=dataset.get_category_sizes('train')
                    )
        
        mlp.load_state_dict(torch.load("./SVG_model/model_" + self.load_as + "_diffusion.pt", weights_only = True))

        print("params loaded")
        
        mlp.to(self.device)    

        self.diffusion = GaussianMultinomialDiffusion(num_classes=K,
                                                      num_numerical_features=num_numerical_features_,
                                                      denoise_fn=mlp,
                                                      gaussian_loss_type=self.gaussian_loss_type,
                                                      multinomial_loss_type=self.multinomial_loss_type,
                                                      parametrization=self.parametrization,
                                                      num_timesteps=self.num_timesteps,
                                                      scheduler=self.scheduler,
                                                      device = self.device)

        
        self.diffusion.eval()
        
        print("diffusion ready")
        
    
        _, empirical_class_dist = torch.unique(torch.from_numpy(dataset.y['train']), return_counts=True)
        # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
#         if disbalance == 'fix':
#             empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
#             x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

#         elif disbalance == 'fill':
#             ix_major = empirical_class_dist.argmax().item()
#             val_major = empirical_class_dist[ix_major].item()
#             x_gen, y_gen = [], []
#             for i in range(empirical_class_dist.shape[0]):
#                 if i == ix_major:
#                     continue
#                 distrib = torch.zeros_like(empirical_class_dist)
#                 distrib[i] = 1
#                 num_samples = val_major - empirical_class_dist[i].item()
#                 x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
#                 x_gen.append(x_temp)
#                 y_gen.append(y_temp)

#             x_gen = torch.cat(x_gen, dim=0)
#             y_gen = torch.cat(y_gen, dim=0)

        # else:
        x_gen, y_gen = self.diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)


        # try:
        # except FoundNANsError as ex:
        #     print("Found NaNs during sampling!")
        #     loader = lib.prepare_fast_dataloader(D, 'train', 8)
        #     x_gen = next(loader)[0]
        #     y_gen = torch.multinomial(
        #         empirical_class_dist.float(),
        #         num_samples=8,
        #         replacement=True
        #     )
        X_gen, y_gen = x_gen.numpy(), y_gen.numpy()
        
        return X_gen, y_gen



