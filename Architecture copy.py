import torch
import os
from agent import *
from collective_functions import *
import torch
import json
import os

#--------------------------------------------------------------------------------------#
#------------------------------- Hyper Paramètres -------------------------------------#
#--------------------------------------------------------------------------------------#

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

#--------------------------------------------------------------------------------------#
#----------------------------------- 5 Agents -----------------------------------------#
#--------------------------------------------------------------------------------------#

num_agents = 5
agents = [DQN_Agent(**hyper_params) for _ in range(num_agents)]
print(f"{num_agents} agents initialized", flush=True)

for i, agent in enumerate(agents):
    checkpoint_path = f"../SVG_model/agent_z1_sent_r100_cf3_{i}"
    if os.path.exists(checkpoint_path):
        if hyper_params["lr_dec"] == 0:
            agent.optimizer.eval()
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path, map_location=device))
        agent.qnetwork_local.eval()
        print(f"Agent {i} model loaded.")
        
#--------------------------------------------------------------------------------------#
#---------------------------------- COVARIABLES ---------------------------------------#
#--------------------------------------------------------------------------------------#

first_layer = agents[0].qnetwork_local.model[0]  # nn.Linear(state_size, layer_size)
weights = first_layer.weight  # [1024, 3280]

feature_importance = weights.abs().sum(dim=0)  # dim=0 => sur les 1024 neurones
topk = torch.topk(feature_importance, k=20)  # top 20 covariables

print("\nTop 20 features les plus importantes :")
for i, (idx, val) in enumerate(zip(topk.indices, topk.values)):
    print(f"{i+1:2d}. Feature index: {idx.item():4d} | Importance: {val.item():.4f}")
