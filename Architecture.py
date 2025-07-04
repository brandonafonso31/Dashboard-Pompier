import torch
import os
from agent import *
from collective_functions import *
import json
import os



os.chdir('/home/brandon/Projet_TER/-Interactive-Emergency-Response-Dashboard-Haute-Garonne-Mike/-Interactive-Emergency-Response-Dashboard-Haute-Garonne-Mike/Scripts')
os.chdir('./Data')
hyper_params = json.load(open("hyper_params.json", "r"))
    
eps_fixed = False
if hyper_params["layer_type"] == "noisy":
    eps_fixed = True
eps = int(not eps_fixed)
    
compute = False

device = torch.device(hyper_params["device"])
torch.autograd.set_detect_anomaly(True)
hyper_params["max_train_steps"] = (530880-1) * 5 # (approx. 5 actions by intervention)
print("max_train_steps", hyper_params["max_train_steps"])
agent = DQN_Agent(**hyper_params)
print("Agent initialized", flush=True)

    
os.chdir('../SVG_model')
print(os.getcwd())
if hyper_params["lr_dec"] == 0:
    agent.optimizer.eval()
agent.qnetwork_local.load_state_dict(torch.load("agent_z1_sent_r100_cf3", map_location="cuda"))
agent.qnetwork_local.eval()
        
print("Eval mode - weights loaded", flush=True)


#--------------------------------------------------------------------------------------#
#---------------------------------- COVARIABLES ---------------------------------------#
#--------------------------------------------------------------------------------------#



first_layer = agent.qnetwork_local.model[0]  # nn.Linear(state_size, layer_size)
weights = first_layer.weight  # [1024, 3280]

# Importance des features (somme des poids absolus sur chaque neurone)
feature_importance = weights.abs().sum(dim=0)  # dim=0 => sur les 1024 neurones

# Extraire les plus importantes
topk = torch.topk(feature_importance, k=20)  # top 20 covariables

print("\nTop 20 features les plus importantes :")
for i, (idx, val) in enumerate(zip(topk.indices, topk.values)):
    print(f"{i+1:2d}. Feature index: {idx.item():4d} | Importance: {val.item():.4f}")
