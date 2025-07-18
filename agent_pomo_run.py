import subprocess
import multiprocessing
from IPython.display import clear_output
import torch
import numpy as np
import os

def run_agent(metrics,agent_id):
    """Fonction pour exécuter un agent individuel"""
    
    #print("\nBien arrivé dans la fonction run_agent !")
    
    suffix = "r100_cf3"
    cmd = [
        "python3", "agent_run.py",
        "--model_name", f"pomo_agent_{metrics}",
        "--hyper_params", "hyper_params.json",
        "--reward_weights", f"rw_{metrics}_{suffix}.json",
        "--dataset", "df_pc_fake_1y.pkl",
        "--start", "1",
        "--end", "53088",
        "--constraint_factor_veh", "3",
        "--constraint_factor_ff", "1",
        "--save_metrics_as", f"agent_pomo_metrics_{metrics}_{suffix}",
        "--train"
    ]
    #print(f"\nLa commande de l'agent pomo_agent_{metrics} est {cmd}\n")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    line_count = 0
    for line in process.stdout:
        line_count += 1
        if line_count % 100 == 0:
            clear_output(wait=True)

        print(line.strip())


def load_agent_rewards(metrics, num_agents=2):
    """Charge les récompenses de tous les agents depuis les fichiers .npy"""
    all_rewards = []
    for i in range(num_agents):
        file_path = os.path(f"Plots/agent_metrics_{metrics[i]}_r100_cf3_train_reward_fake.npy")
        if os.path.exists(file_path):
            rewards = np.load(file_path)
            all_rewards.append(rewards)
    return np.array(all_rewards)

def compute_pomo_loss(all_rewards):
    """Calcule la loss POMO avec baseline partagée comme dans le papier"""
    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
    baseline = rewards_tensor.mean(dim=0, keepdim=True)
    advantages = rewards_tensor - baseline
    loss = -torch.mean(advantages * torch.log(rewards_tensor + 1e-10))
    return loss


if __name__ == "__main__":
    
    #print("\nBien arrivé dans agent_pomo_run !\n")
    num_agents = 2 # Nombre d'agents POMO
    print(f"Nombre d'agent = {num_agents}. on démarre le multi process ...")
    
    metrics = ['z1_sent', 'v_not_found_in_last_station']
    # , 'ff_sent', 'ff_skill_mean', 'v_degraded']
    # , 'function_not_found', 'skill_lvl', 'function_cancelled', 'cancelled', 'v_sent_full', 'v_sent', 'v1_not_sent_from_1st_station']
  
    tasks = [(metric, i) for i, metric in zip(range(num_agents), metrics)]
    #print(tasks)
    with multiprocessing.Pool(processes=num_agents) as pool:
        pool.starmap(run_agent, tasks)
        
    all_rewards = load_agent_rewards(metrics, num_agents)
    loss = compute_pomo_loss(all_rewards)