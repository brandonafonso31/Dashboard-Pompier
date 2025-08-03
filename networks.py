import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from copy import copy




# def weight_init(layers):
#     for layer in layers:
#         torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
# def weight_init(layers):
#     for layer in layers:
#         if isinstance(layer, nn.Linear):
#             nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
#             if layer.bias is not None:
#                 nn.init.zeros_(layer.bias)
def weight_init(modules):
    for module in modules:
        if isinstance(module, nn.Sequential):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
def weight_init_xavier(layers):
    for layer in layers:
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)

class NoisyLinear(nn.Linear):
    # Noisy Linear Layer for factorised Gaussian noise 
    def __init__(self, in_features, out_features, sigma_init=0.5, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_init = sigma_init
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_weight = nn.Parameter(torch.FloatTensor(out_features, in_features))

        # extra parameter for the bias 
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features,))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        # not trainable tensor for the nn.Module
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))
        # reset parameter as initialization of the layer
        self.reset_parameter()
    
    def reset_parameter(self):  

        bound = 1 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def f(self, x):

        return x.normal_().sign().mul(x.abs().sqrt())

    def forward(self, input):
        # sample random noise in sigma weight buffer and bias buffer
        self.eps_p.copy_(self.f(self.eps_p))
        self.eps_q.copy_(self.f(self.eps_q))

        weight = self.mu_weight + self.sigma_weight * self.eps_q.ger(self.eps_p)
        bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()

        return F.linear(input, weight, bias) 
    
class Dueling_QNetwork(nn.Module):

    def __init__(self, state_size, action_size,layer_size, n_step, seed, num_layers = 8, layer_type="ff", use_batchnorm=True):

        super(Dueling_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.use_batchnorm = use_batchnorm
        

        layers = []

        layers.append(nn.Linear(state_size, layer_size))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        
        if layer_type == "noisy": # pas de batchnorm pour les noisy nets

            self.ff_1_A = NoisyLinear(layer_size, layer_size)
            self.ff_1_V = NoisyLinear(layer_size, layer_size)
            self.advantage = NoisyLinear(layer_size,action_size)
            self.value = NoisyLinear(layer_size,1)
            weight_init([self.model,self.ff_1_A, self.ff_1_V])

        else:

            self.ff_1_A = nn.Linear(layer_size, layer_size)
            self.ff_1_V = nn.Linear(layer_size, layer_size)
            self.bn_A = nn.BatchNorm1d(layer_size)
            self.bn_V = nn.BatchNorm1d(layer_size)
            self.advantage = nn.Linear(layer_size,action_size)
            self.value = nn.Linear(layer_size,1)
            weight_init([self.model,self.ff_1_A, self.ff_1_V])       
        
    def forward(self, state):
        # print("state:", state.shape)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = self.model(state)
        # print("post unsq state:", state.shape)

        if self.layer_type == "noisy": # pas de batchnorm pour les noisy nets
            x_A = torch.relu(self.ff_1_A(x))
            x_V = torch.relu(self.ff_1_V(x))
        else:
            x_A = torch.relu(self.bn_A(self.ff_1_A(x)))
            x_V = torch.relu(self.bn_V(self.ff_1_V(x)))

        value = self.value(x_V)
        value = value.expand(state.size(0), self.action_size)
        advantage = self.advantage(x_A)
        Q = value + advantage - advantage.mean()
        if self.use_batchnorm:
            Q = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            Q = value + advantage - advantage.mean()
        return Q



class QVN(nn.Module):
    """Quantile Value Network"""
    def __init__(self, state_size, action_size,layer_size, n_steps, device, seed, noisy, N):
        super(QVN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.N = N
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(1, self.n_cos+1)]).view(1,1,self.n_cos).to(device)
        self.device = device
        if noisy:
            layer = NoisyLinear
        else:
            layer = nn.Linear


        # Network Architecture
 
        self.head = nn.Linear(self.state_size, layer_size) 
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = layer(layer_size, layer_size)
        self.cos_layer_out = layer_size
        if not noisy: weight_init([self.head, self.ff_1])
        self.advantage = layer(layer_size, action_size)
        self.value = layer(layer_size, 1)
        if not noisy: weight_init([self.ff_1])


    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]
        
    def calc_cos(self,taus):

        batch_size = taus.shape[0]
        n_tau = taus.shape[1]
        cos = torch.cos(taus.unsqueeze(-1)*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos
    
    def forward(self, input):

        return torch.relu(self.head(input))
        
    def get_quantiles(self, input, taus, embedding=None):

        if embedding==None:
            x = self.forward(input)

        else:
            x = embedding
        batch_size = x.shape[0]
        num_tau = taus.shape[1]
        cos = self.calc_cos(taus) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.cos_layer_out) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.cos_layer_out)   
        x = torch.relu(self.ff_1(x))
 
        advantage = self.advantage(x)
        value = self.value(x)
        out = value + advantage - advantage.mean(dim=1, keepdim=True)

        return out.view(batch_size, num_tau, self.action_size)
    
    

class FPN(nn.Module):
    """Fraction proposal network"""
    def __init__(self, layer_size, seed, num_tau=8, device="cuda:0"):
        super(FPN,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_tau = num_tau
        self.device = device
        self.ff = nn.Linear(layer_size, num_tau)
        self.softmax = nn.LogSoftmax(dim=1)
        weight_init_xavier([self.ff])
        
    def forward(self,x):


        q = self.softmax(self.ff(x)) 
        q_probs = q.exp()
        taus = torch.cumsum(q_probs, dim=1)
        taus = torch.cat((torch.zeros((q.shape[0], 1)).to(self.device), taus), dim=1)
        taus_ = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        
        entropy = -(q * q_probs).sum(dim=-1, keepdim=True)
        assert entropy.shape == (q.shape[0], 1), "instead shape {}".format(entropy.shape)
        
        return taus, taus_, entropy
    
class POMO_Network(nn.Module):
    def __init__(self, state_size, action_size, layer_size, num_layers, use_batchnorm, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        # Construction dynamique des couches
        layers = []
        input_dim = state_size
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, layer_size),
                nn.ReLU()
            ])
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(layer_size))
            input_dim = layer_size  # Les couches suivantes prennent layer_size en entrée

        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Linear(layer_size, action_size)  # Couche de sortie

    def forward(self, x, mask_batch=None):
        """
        Args:
            x: [B, state_size]
            mask_batch: [B, action_size] or None
        Returns:
            logits: [B, action_size]
        """
        x = self.encoder(x)  # [B, hidden_dim]
        logits = self.decoder(x)  # [B, action_size]

        if mask_batch is not None:
            logits = logits + mask_batch  # ajouter -inf aux actions interdites

        return logits